# Learning Robust Intervention Representations with Delta Embeddings

- Decision: Accept (Poster)
- Scores: 6, 4, 6, 8

## Abstract
Causal representation learning has attracted significant research interest during the past few years, as a means for improving model generalization and robustness. Causal representations of interventional image pairs (also called ``actionable counterfactuals'' in the literature), have the property that only variables corresponding to scene elements affected by the intervention / action are changed between the start state and the end state. While most work in this area has focused on identifying and representing the variables of the scene under a causal model, fewer efforts have focused on representations of the interventions themselves. In this work, we show that an effective strategy for improving out of distribution (OOD) robustness is to focus on the representation of actionable counterfactuals in the latent space.  Specifically, we propose that an intervention can be represented by a Causal Delta Embedding that is invariant to the visual scene and sparse in terms of the causal variables it affects. Leveraging this insight, we propose a method for learning causal representations from image pairs, without any additional supervision. Experiments in the Causal Triplet challenge demonstrate that Causal Delta Embeddings are highly effective in OOD settings, significantly exceeding baseline performance in both synthetic and real-world benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes Causal Delta Embeddings (CDE), a method to learn robust representations of interventions from image pairs. The key idea is to represent an intervention as the difference between pre- and post-intervention latent embeddings. The method combines cross-entropy, supervised contrastive, and sparsity losses to enforce independence, sparsity, and invariance. Results on the Causal Triplet benchmark show strong out-of-distribution generalization and meaningful representation structure.

### Strengths
- Clear motivation and solid theoretical grounding.
- Simple but elegant framework using delta embeddings to capture interventions.
- Strong experimental performance.
- Well-written and well-structured paper, with convincing ablations and visualizations.

### Weaknesses
- The method is trained with three losses. It is unclear how sensitive the performance is to the weighting between them. Although the appendix provides implementation details, it is not clear how one would set these weights in practice.

### Questions
See weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes to learn representations corresponding to interventions, in contrast to learning representations for the underlying variables. The motivation behind this is to learn reusable representations/embeddings for interventions invariant of the object on which the intervention acts. From the paper, I gather that such representations are useful in Visual Language Action (VLA) models.

### Strengths
The idea of learning representations for actions is interesting, and it is a reasonable choice to learn these representations using interventional data. I believe Causal Delta Embeddings (CDEs) will have practical applications in robotics and other interactive domains.

### Weaknesses
I include my major concerns under "weaknesses" and minor concerns (mainly related to writing) under "questions." My most important concerns are W1, W2, and W5 (d, e). Most weaknesses/questions can be answered without experiments. I will raise my score if my concerns are addressed.

**W1. Interventional vs counterfactual**: A major confusion I have is whether CDE requires interventional or counterfactual data, the latter being a stricter requirement. In lines 194-197, the goal is stated to learn some function using a dataset of pre- and post-interventional data. If these data samples are obtained with the same exogenous noise, they are counterfactual samples. The equation in lines 157-158 mentions the exogenous noise $\epsilon$, but its role in the training dataset is unclear. "Identical noise" is assumed in lines 211, which implies the requirement is counterfactual data. To satisfy eq. (1) using interventional data, the encoder $\phi$ must disregard the exogenous noise completely. Can the authors clarify what the real requirements are?

**W2. Should the underlying representation be identifiable?** Even if counterfactual data is provided, eq. (1) can be satisfied only using an identifiable $\phi$. Is my understanding correct? If that's the case, will such an identifiable encoder $\phi$ be automatically learned while learning CDE, or is it a starting requirement? If my understanding is wrong, please provide a counterexample and include it in the text. I have a related concern on line 216. It is written "an action's representation is independent of the causally irrelevant..." Does "independent" here mean statistically independent, or that it is not affected by the "causally irrelevant elements?"

**W3. Will the aggregation module allow patch-level CDE?** In Sec. 5.2.1, the loss function is applied only over the aggregated embeddings. It is possible for patch-level embeddings to not follow Def. 2 even when the aggregated embedding satisfies Def. 2. Is that true? Are patch-level CDEs not required to follow Def. 2?

**W4. Comparison to BISCUIT**: BISCUIT (Lippe et al., 2023) also encodes the action variable responsible for interventions. See Fig. 7 in (Lippe et al., 2023). Is it possible to compare CDE against BISCUIT?

**W5. Questions on experiments**:

**W5. (a)** What are the baseline models trained on? Lines 257-259 say that the encoder is a DINO-pretrained vision backbone. Are ResNets and ViT in Tables 1 and 2 also DINO-trained? Why is it compared against a CLIP in Table 2, instead of another DINO? What exactly is the oracle-mask approach?

**W5. (b)** How are object-centric models, such as Slot Attention, adapted to predict actions?

**W5. (c)** Why is OOD Comp. accuracy smaller than OOD Syst. accuracy for all baselines, except CDE in Table 1? OOD Syst. seems to be a more difficult task.

**W5. (d)** How is CDE able to achieve that much accuracy in OOD Syst. in Table 1? The manifestations of actions in images are linked to the object on which it acts. So how can the model foresee what the action will look like on an unseen object? I can think of a possibility: CDE works only in scenarios where the action manifestation on the unseen object was seen during training, and that can also maybe explain the difference between OOD Comp. and OOD Syst. accuracies. Can the authors explain why CDE works on OOD Syst.?

**W5. (e)** The CE-only model in Table 3 beats all baselines in Table 1. I thought vanilla-R and vanilla-V were also trained with just CE. So does CDE benefit from something beyond its loss functions?

**W5. (f)** What is Fig. 10 supposed to convey? If the point was to show that CDEs are suitable for k-NN, then it must be compared with other baselines or other variants of CDE with fewer losses (like in Table 3). Although I would appreciate answering this question with experiments, I will not reduce my score if the experiments are not provided, as it is not a main experiment.

### Questions
**Q1. Related works on CRL from interventional data**: In related works, some works on causal representation learning (CRL) that use interventional data are mentioned. However, these works are not mentioned in the introduction when the story is built around the existing state of CRL. I think it is important to highlight the current advances in CRL using interventional data, and how this work is different from them. Also, I would suggest adding more recent works in CRL using interventional data. See [A1-3] and the references therein. [A1] is a contemporary work. The authors need not include it, but I shared it here as a source for recent works.

**Q2. Related works on contrastive learning**: There are two sentences on contrastive learning -- one listing two general contrastive learning works, and another providing a slightly unfair comparison w.r.t. this work. While contrastive learning only compares individual samples, it also does not require any pre- and post-intervention sample pairs, like this work. Another thing is that several works that link contrastive learning to CRL are missing. Some of them are [A4-5] and related works in [A6].

**Q3. Related works on SMS**: Again, a few important works on SMS are missing from your literature survey. See [A7-9].

**Q4. Minor writing comments**:

**Q4. (a)** I suggest rewording the sentence in line 121, starting with "While previous methods..." to clarify what part of the interventional mechanism is captured by CDEs and what invariance (across contexts) is targeted by CDE.

**Q4. (b)** Line 154: "a set of causal variables $Z\in\mathcal{Z}\subset\mathbb{R}^l$..." Which variable is the set here?

**Q4. (c)** Line 161: "a complex, **non-invertible** generative function..." How can you learn actions if the variables on which the action works are not retrievable?

**Q4. (d)** How did eq. (3) come about? Is $f$ in eq. (3) same as $f$ in line 157? How did the second equality in eq. (3) come about? Are the non-zero indices of $\delta_a$ aligned with the changes in $z_a$ due to the action?

**Q4. (e)** Can you clarify what the sentence in line 323 starting with "If, however, these..." means?

**Q4. (f)** I find lines 425-427 to be slightly misleading. In Fig. 8, there are indeed some anti-parallel representations learned. But there are also some nearly-anti-parallel representations for action pairs that do not make sense. For example, cut-break, move-pull and move-eat have around -0.75 in Fig. 8.

**Q4. (g)** Do any of the numbers in Tables 5 and 6 appear anywhere in the main results?

**References**

[A1] Pranamya Kulkarni, Puranjay Datta, Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, Ali Tajer, "ROPES: Robotic Pose Estimation via Score-Based Causal Representation Learning", ArXiv 2025.

[A2] Burak Varıcı, Emre Acartürk, Karthikeyan Shanmugam, Ali Tajer, "Linear Causal Representation Learning from Unknown Multi-node Interventions", NeurIPS 2024.

[A3] Dingling Yao, Dario Rancati, Riccardo Cadei, Marco Fumero, Francesco Locatello, "Unifying Causal Representation Learning with the Invariance Principle", ICLR 2025.

[A4] Julius von Kügelgen, Yash Sharma, Luigi Gresele, Wieland Brendel, Bernhard Schölkopf, Michel Besserve, Francesco Locatello, "Self-Supervised Learning with Data Augmentations Provably Isolates Content from Style", NeurIPS 2021.

[A5] Roland S. Zimmermann, Yash Sharma, Steffen Schneider, Matthias Bethge, Wieland Brendel, "Contrastive Learning Inverts the Data Generating Process", ICML 2021.

[A6] Dingling Yao, Danru Xu, Sébastien Lachapelle, Sara Magliacane, Perouz Taslakian, Georg Martius, Julius von Kügelgen, Francesco Locatello, "Multi-View Causal Representation Learning with Partial Observability", ICLR 2024.

[A7] Elliot Layne, Jason Hartford, Sébastien Lachapelle, Mathieu Blanchette, Dhanya Sridhar, "Sparsity regularization via tree-structured environments for disentangled representations", ArXiv 2024.

[A8] Sébastien Lachapelle, Pau Rodríguez López, Yash Sharma, Katie Everett, Rémi Le Priol, Alexandre Lacoste, Simon Lacoste-Julien, "Nonparametric partial disentanglement via mechanism sparsity: Sparse actions, interventions and sparse temporal dependencies", ArXiv 2024

[A9] Danru Xu, Dingling Yao, Sébastien Lachapelle, Perouz Taslakian, Julius von Kügelgen, Francesco Locatello, Sara Magliacane, "A Sparsity Principle for Partially Observable Causal Representation Learning", ICML 2024.

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
3

### Summary
The paper introduces the concept of delta embeddings to represent atomic causal interventions between two images (e.g. open or close a drawer on the image) and proposes a simple method for classifying intervention actions. The proposed method first separately generates embeddings of a pair of images distant by a single intervention using a backbone Vit coupled with a MLP head tasked to disentangle the embeddings into delta embeddings. Then, the difference between the two delta embeddings is computed and sent as an input to an action classifier. The pipeline is trained end-to-end and regularized with contrastive and sparsity losses to improve the creation of robust representations that hold out-of-distribution. The method achieves improved o.o.d performance compared to baselines and generates embeddings with meaningful relationships between classes of actions.

### Strengths
1. The paper tackles a challenging problem in causal representation learning, namely the disentanglement of interventions, using a an original approach. The geometry of Delta embeddings could potentially convey very meaningful information, as hinted by the experiments and visualizations in the appendix.
2. The paper is well-written and easy to understand. The theoretical section complements well the description of the approach, justifying it accurately.
3. The experiments on out-of-distribution splits are particularly useful for assessing the generalization of the proposed delta embeddings.

### Weaknesses
Experiments are conducted on a single benchmark (causal Triplet), which limits the generalizability of the findings (although the experiments in o.o.d settings mitigate this issue). Using larger backbone models on more datasets would further strenghten the contributions.

### Questions
1. How realistic is the assumption of independence of latent factors in the data generative process? Indeed, the presence or absence of an object can have an effect on the lighting of the scene or on objects on top of it.
2. Must the set of possible actions $\mathcal{A}$ be known in advance or can the approach generalize to new actions, e.g. compositions of actions?
3. Have you conducted experiments with additional backbone models, e.g. ResNet-18, for a fair comparison with other baselines?
4. How do you interpret the drop in performance in o.o.d in Table 2 for the procTHOR multi-object setting?

### Soundness
3

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
4

### Summary
This paper tackles the task of learning robust representations via interventional data. It proposes Causal Delta Embeddings (CDE) for representing interventions as latent-space deltas between pre/post-states. Results on synthetic and real benchmarks show gains in triplet evaluations and effective identification of changed semantics.

### Strengths
* tackles a very important problem of learning robust and interpretable representations 
* leverages pretrained vision encoders in a good way and moves away from toy-dataset-only evaluations. 
* clear conceptual framing of intervention representation problem
* strong quantitative OOD gains; well-executed ablations
* visualization & semantic structure analysis support claims

### Weaknesses
* only evaluates one vit backbone
* requires heavy supervision that is only possible with synthetic data
* empirical gains limited in real-world 
* lacks exploration of confounding effects or imperfect interventions

### Questions
* how robust is CDE to noise or partial observability in interventions?
* does sparsity regularization risk collapsing subtle but real effects?
* how does the performance change when using different pretrained models? MAE, VQ-VAE's encoder, CLIP  would be very interesting

### Soundness
4

### Presentation
3

### Contribution
4
