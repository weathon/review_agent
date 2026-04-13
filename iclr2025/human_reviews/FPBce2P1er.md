## Human Reviewer 1

### Summary
This paper considers under what circumstances a kernel model learning on top of a disentangled representation can generalize compositionally. The main finding is that these models can generalize compositionally if the task is what the authors call conjunction-wise additive with respect to the disentangled features. Although the theory is developed using kernel models, most of the results generalize more or less to deep neural networks, which in some training regimes are related to kernel models. Overall I really liked this paper. The results are interesting and novel, the paper is for the most part clear and well-written, and the results help push the field forward in terms of designing models which are more likely to generalize compositionally.

### Strengths
1. The reason for looking at kernel models is very clearly articulated and motivated.
2. The theory is supplemented with good empirical experiments that show it applies in deep neural networks, even though it was developed with kernel models (with a nice method for deep learn-afying the main tasks used in the kernel models sections).
3. Contributions and advantages over prior work are very clear (e.g., no assumptions about modularity or linear/additive combinations in the downstream model).
4. The practical consequences of the main finding, laid out in Section 4.3, are clearly articulated with several simple examples.
5. The work helps guide the design of more difficult compositional generalization benchmarks that cannot be solved by kernel models (or neural networks trained in the kernel regime).
6. Figures are clear and aesthetic, both when illustrating the framework/tasks and when presenting results.
    1. One exception I recommend fixing is Figure 3d, where the borders of the bar are too thick and obscure the colors of the bars.

### Weaknesses
1. The paper discusses compositional generalization with respect to compositional representations, but equates compositional representations with disentangled representations. I think it’s fair to say that other compositional structures are possible in additional to disentanglement (e.g., Tensor product representations, block-wise disentanglement, etc.), and disentanglement is likely just one particular special case of compositional representations. Ideally I would want to see the whole paper rewritten in terms of disentanglement instead of compositionality (disentangled representations, generalization from disentangled representations). I understand that the authors might not want to go this far, so in this case I would make it very clear in the introduction that you are only considering disentangled representations, and that these are a subset of what might be considered compositional representations more generally.
2. At times it is unclear whether this work is about compositional tasks (tasks where the optimal solution would necessitate some compositional representation/function), compositional inputs/datasets, or compositional representations of inputs. All three are used at various times, which leads to confusion about the scope of what is being studied, especially in earlier parts of the paper before section 3.1 where it becomes clear that $z$ is the disentangled representation which would have been extracted in some earlier part of a model, like an intermediate layer in a neural network. Even after 3.1, though, at times “input” and “representation” are used interchangeably, which causes confusion. This extends even into the discussion section (”dataset statistics and compositional generalization” as opposed to “representation statistics and compositional generalization”).
3. The sections introducing kernel regime and kernel model, in particular in Sections 2 and 3.2, can be cleaned up. Overall, I suggest a restructuring of the content regarding kernel models and your setup. After the introduction, I would suggest immediately introducing your setup and defining variables, as is done currently in Section 3.1, and then describing kernel models within that framework. While doing this, keep a concrete running example that makes it clear what $x$, $z$, $\phi$, and $K$ would refer to on some task that readers would be familiar with. The related work can come after this setup, so that we can make a connection between the related work and your particular setting. Below are some other stray comments about confusions during the introduction of kernel models in your setup.
    1. At times $x$ is used and at other times $z$ is used. Are these the same? Is it a typo in Section 3.2 for instance where you write $f_w(x)$ as opposed to $f_w(z)$?
    2. In equation (1), where do the dual coefficients come from? Are they determined from $w$ and $\phi$ in some way?
    3. Are you making a distinction between “input” and “representation”? Which of these is disentangled in your theory, $z$ or $\phi(z)$? At times, especially earlier on in the paper like in Section 2, it is not obvious.
4. Section 3.1 says that the target $y$ is given by an arbitrary function of $z$ and that your framework is agnostic to this function. This is a bit misleading, as the core contribution of the work is to formalize what downstream functions kernel models can compositionally generalize to w.r.t. some observed disentangled features.
5. Section 4.2 defines conjuction-wise additive functions, but it is quite difficult to quickly parse the math and get intuition for it. Intuition for the proof is given, but not intuition for what a conjunction-wise additive function is. This is a shame because after spending enough time to digest the definition, it is intuitively quite simple. Please try and provide more helpful intuition, as well as an example of a conjunction-wise additive function and how it differs from an additive function.
6. The theoretical and empirical results in Section 5.1 seem to be of significant consequence. Specifically, Proposition 5.1 seems to suggest that very deep neural networks in the kernel regime are unlikely to generalize compositionally as they only represent the full conjunction of disentangled features (if I understand correctly, another way of stating this is that they memorize the training data). While this is explored further in the subsequent subsections, the immediate consequences of Proposition 5.1 are only unpacked in a single sentence following the proposition. I think this result should be emphasized more, and Section 5.1 should do a better job of foreshadowing/leading the results in subsequence subsections.
    1. Additionally, I don’t think it will be completely obvious to everyone why assigning weight to the full conjunction function amounts to memorization. Maybe it can be more clearly articulated (e.g., the full conjunction is a function of combinations of features that can amount to something like a lookup table, without any ability to generalize).
7. Small point: I recommend citing Schug 2024 http://arxiv.org/abs/2312.15001 at the same place as where you cite Wiedemer 2023.

### Questions
1. In Section 3.1 first paragraph, the definition of the disentangled representation is a bit unclear.
    1. Does each component have a finite set of possible values it can take on (e.g., multiple values for the “color” component)?
    2. Are the different possible values within a component orthogonal (e.g., vectors for different colours), or are only the vectors across components orthogonal (e.g., colour and shape representations existing in different subspaces)?
    3. Is $C$ constant across samples? In other words, do the individual components (like color and shape) always apply to each possible input?
2. In Section 5 and subsections within, why is the approach of looking at the kernel and representational salience referred to as “modular”?
3. Section 5.2 refers to a set $\mathcal{W}$, but I can’t seem to find where the meaning of this variable was defined earlier on in the paper. Apologies if I’ve missed it, but what is $\mathcal{W}$?
4. See above weaknesses for other questions.

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
10

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper theoretically analyzes how the model can generalize downstream tasks when the ground-truth disentangled representations are achievable. The authors prove that different models can generalize well on component-wise additivity tasks. Meanwhile, the models are limited in learning tasks involving conjunction-wise additivity, which makes them hard to generalize. Based on the analysis, the authors figured out two important failure modes, i.e., memorization leakage and shortcut bias. The former is because the model learns too many non-atom features from the training set while the latter is caused by the hidden spurious correlation in the training data. Finally, the authors empirically verify their theory using the behavior of some deep neural network structures. Although the paper studies a very important and novel problem about compositional generalization ability, I find it is a struggle for me to understand the paper in depth. However, I do believe the assumption that perfect disentangled representations are achievable harms the applicability of the paper’s results. So I tend to give a negative evaluation.

### Strengths
The paper studies an important question that might be overlooked by many related works in compositional generalization, i.e., when perfect disentangled representations are given, what types of downstream tasks are solvable (or non-solvable) in kernel methods? The paper theoretically analyzes this problem and verifies its theory using different experiments.

### Weaknesses
### Weakness:

My main concern in this part is about the gap between theory (with many assumptions) and practical systems. Although the paper proposes several experimental results to support the theory, the following concerns forbid me from giving a positive evaluation of this paper:

- The gap between kernel method assumption and deep neural networks. Analyzing the model’s behavior using a simplified model is acceptable, but I expect more results to show that the gap between theory and practical models is negligible. For example, will the deep network on real image input also have a salience score similar to Figure 2? Can we design an ablation study that directly manipulates the weights of overlapping features, e.g. $f_{12}$, and see its influence on the generalization ability? Although the answer might or might not support the claims, knowing the limits of the proposed theory is beneficial.
- The paper assumes that perfect disentangled representations are accessible, which could be easily violated in practical scenarios. Then, how will the results change when the representations are partially disentangled? If the theoretical analysis under this more practical condition is hard, some experimental results under a non-perfect case would be helpful.
- In section 6, the authors replace the one-hot input with real images, which is good. However, the task is still not so practical. Considering the experiments on more common compositional generalization problems, e.g., the dsprite dataset in [1],  can make the claim stronger.
- The paper has potential and the author did a good job of formalizing an important problem. However, the connection between the experimental part and the theory is still not quite clear. I think adding a high-level summary or conceptual diagram that ties together the main theoretical and empirical contributions of the paper would make it easier to understand the whole story of the paper.

[1] Xu, Zhenlin, Marc Niethammer, and Colin A. Raffel. "Compositional generalization in unsupervised compositional representation learning: A study on disentanglement and emergent language." Advances in Neural Information Processing Systems 35 (2022): 25074-25087.

### Questions
- In Figure-1b, why [-2]+[1]=1?
- The paper is a bit abstract for me to understand in depth. Especially, I cannot link the provided theory to practical applications, although section 6 indeed provides some results about deep neural networks and common datasets. I think the presentation of the paper is generally good: in the first several sections, we learn how the compositional generalization task and the kernel model studied in this paper are defined. We also learned the salience score is the main metric for tracking different types of features (e.g., how many ground truth features that this learned feature depends on). With this measurement, we know from proposition 5.1 that the later part of the network prefers using those higher-order features, which match our intuitions well. Then, we learn that memorization leak, which could be captured by those higher-order features, hinders the model’s generalization. After that, I began to feel confused: how could we link the results in Figure 4 to the theory provided in the previous part? I think a more detailed analysis of how results in section 6 are correlated to different parts of the theory (e.g., which proposition, which claim, etc.) would make the paper clearer.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper considers the question, "Given a compositional (i.e., disentangled) representation, under what circumstances can kernel models (e.g., a linear readout trained with SGD) generalize compositionally?" The usefulness of disentangled representation for compositional generalization has been debated in the literature, but generally from an empirical angle or specific settings (e.g., object-centric learning). The authors opt to study the problem theoretically from the perspective of kernel models. This approach leads to a reasonably general framework that highlights important limitations: The type of computation that can be generalized is restricted to simple additive combinations of components, what the authors call "conjunction-wise additive". Even within this class of tasks, the authors go on to show that a kernel model will often not generalize perfectly, as it is biased to consider (spurious) interactions between all components or fall for spurious shortcuts.

### Strengths
This paper presents a comprehensive study of compositional generalization. While the setting is restricted to fixed, disentangled representations (which, in practice, are not guaranteed to be recovered by the model), it succeeds in abstracting compositional generalization to encompass many specific problem formulations. The focus on kernel methods allows the authors to draw insightful conclusions about the limits of compositional generalization in terms of the operations that can be learned, which are entirely novel to the best of my knowledge. It is nice to see that the biases predicted by the theory can be demonstrated on real (if toyish) datasets.

### Weaknesses
The paper is quite dense and not easily accessible to readers. I can see that the authors attempted to illustrate their theoretical findings on a few example tasks to convey some intuition, but the explanation of the example tasks is still frequently very technical and hard to parse. For example, it is still somewhat unclear to me why transitive equivalence is ruled out by Thrm. 4.1 (see also minor suggestions below).

Given Thrm. 4.1, I would have appreciated a short discussion on existing compositional generalization tasks from the literature, e.g., from Schott et al., or Locatello et al., or Hupkes et al. Which of these tasks would be solvable by a kernel model? Could the insights from this paper aid in understanding why the existing literature on the usefulness of disentangled representations is so divided?

### Questions
# Questions
- I'm not sure Wiedemer et al. (2023b) is characterized correctly as requiring a network that is constrained to be linear/additive. If I recall correctly, this paper allowed for arbitrary combinations of components. If so, a more detailed comparison of the assumptions in that work would be helpful to assess the new insights from this work.
- LL269: Why does $f_{12}(z)$ "fall away"? I assume because $f_J(z_{12}) = 0$, but why? Could you elucidate this on a brief example?
- §5.1 / §B: Even after reading these sections, it is unclear to me how the representational salience is computed in practice. Could you walk me through an example with a batch of training points?
- §5.1 / Fig. 2: Do you have any intuition where the difference between nonlinearities is coming from?
- §6: Am I correct in assuming all models are randomly initialized and not trained? This is not entirely clear from the text. Also, what specific model architectures were used, and how are they initialized? What are the training settings for the linear readout? As the code will not be published, these details are essential to ensure reproducibility. I recommend adding a corresponding section detailing initialization, architecture, and other specifications to the main text or appendix.

# Minor suggestions
- LL99: Why call a combination of components a "conjunction", and not the more obvious "composition"?
- Fig. 1c: It was not immediately clear which side of the training set line is the training/test set
- LL256: "The readout weight ..." is unclear, is there a word missing? When will it change?
- L269: should be "test set"
- LL317: reusing $c$ here as a number of components is slightly confusing when $c$ was used above as a component index. I recommend using $k$ or $n$ instead
- LL354: What is $\mathcal W$ here? It has not been introduced before

### Soundness
4

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
This paper studies compositional generalization from the perspective of kernel theory, showing that they are constrained to adding up values assigned to each combination seen during training. This result demonstrates a fundamental restriction in the generalisation capabilities of kernel models. The theory is validated empirically and shows, that the theory also captures certain behaviors of deep neural networks trained on compositional tasks.

### Strengths
The work tries to establish a stronger theory of compositional generalization in kernel and neural network models, addressing a crucial white spot in our understanding of statistical machine learning models. It also takes on an original angle to the problem by studying compositionality in the kernel limit, and I appreciate the empirical evaluation and comparison w.r.t. to deep neural networks.

### Weaknesses
- There is a wide gap between the claims of the paper and what is actually shown and proved. In particular, the theory only considers kernel models with random weights networks as feature extractors. In this limit the intermediate representation is compositionally structured (assuming the input is as well). But that doesn't necessarily extend to other kernels as implied by the abstract, which doesn't mention this limitation once.

- Also, while the work tries hard to make a connection to disentangled networks and thus establish an empirical relevance (which I generally welcome), the paper very carefully states that "This highlights fundamental computational restrictions on [...] pretrained models with disentangled representations that are fine-tuned in the kernel regime and infinite-width neural networks". For one, that's overly broad because even if a DNN is trained on disentangling the representation, there is no guarantee it is disentangled outside of the training data (in contrast to what is considered in the theoretical results). Second, this only applies if also the input is already compositional, rendering the statement mostly irrelevant for any practical application (e.g., in vision or language you don't have a compositional input to start with).

- The overall presentation is quite confusing and hard to follow and I am having a hard time even understanding what the precise contribution of the paper is, what assumptions are being made and what conclusions can be drawn from it. In particular, what is drawn from the results is mixed up with the precise theoretical results. It would be great to have one section that is really precise and doesn't immediately tries to generalise to pre-trained models, before outlining precisely how and in what way the results might generalise to empirical models and tasks.

- There are pretty much no details on the empirical experiments with deep neural networks (both in the main text and the appendix). That part is thus almost impossible to evaluate. However, if my assumptions are correct, then the DNNs are trained in extremely simplistic settings that resemble nothing even close to what the networks are actually used for in practice (despite the use of MNIST or CIFAR images). Hence, claiming that the theory can explain the empirical behavior of DNNs is misleading.

### Questions
- Line 262 states that "kernel models cannot solve any task that cannot be expressed in a conjunction-wise additive terms" - can you explain which tasks cannot be phrased in this way? In the worst case, the full conjunction can express any non-linear relationship, no?

- Line 269: why would the term f_{12}(z) fall away on the test set? I don't see how this should happen.

- How can a compositionally structured kernel model learn arbitrary training data, as is implied by section 4.3?

- Please add all the details on your empirical evaluation.

- In the related work, you state that Wiedemer et al. and Lachapelle et al. is restricted to linear networks, but you probably mean linear interactions between the components.

- Regarding the conjunction-wise additive computation: Why is the result surprising? In line 464 it states that "neural networks tend to implement conjunction-wise additive computations - at least when trained on conjunction-wise additive tasks". It would be quite surprising if it wouldn't implement that, no?

My scores currently reflect that the contributions of the paper are very difficult to assess and that the claims are way too broad (at least that's my current understanding). I can see interesting aspects and would like to encourage the authors to outline and state their contributions very clearly and without overclaiming. Understanding generalisation in machine learning is hard, so even some understanding on random-weights models is definitely interesting.

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
3

### Confidence
3