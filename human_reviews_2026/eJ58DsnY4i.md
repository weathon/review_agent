# A Neuro-symbolic Approach to Epistemic Deep Learning for Hierarchical Image Classification

- Avg Score: 1.50
- Decision: Reject
- Scores: 0, 2, 2, 2

## Abstract
Deep neural networks achieve strong recognition performance, but they often produce overconfident predictions and fail to respect structural constraints in data. We propose a neuro-symbolic framework that augments Swin Transformers with focal set reasoning and differentiable fuzzy logics. Rather than treating labels as isolated categories, the model induces focal sets by modelling overlaps in the learned embedding space, which helps capture epistemic alternatives beyond single labels. These focal sets form the basis of a belief-theoretic layer that uses fuzzy membership functions and $t$-norm conjunctions to encourage consistency between fine- and coarse-grained predictions. A learnable loss further balances calibration, mass regularisation, and logical consistency, allowing the model to adaptively trade off symbolic structure with data-driven evidence. In experiments on hierarchical image classification, our framework maintains accuracy on par with transformer baselines while providing more calibrated and interpretable predictions, reducing overconfidence, and enforcing high logical consistency across hierarchical outputs. Overall, our results suggest that combining focal set reasoning with fuzzy logics provides a practical step toward deep learning models that are both accurate and epistemically aware.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper studies uncertainty-aware learning with neuro-symbolic models. In particular, it suggests applying subjective logic to high-level prediction of a hierarchical clustering setup. Combining focal set reasoning and differentiable fuzzy logic, the paper arrives at a new loss function that can be dropped into a feed-forward prediction pipeline. The goal is to improve calibration a more interpretable way than the existing methods.  The suggested approach has been evaluated on a transformer variant, the Swin transformer, and tested on two standard hierarchical classification benchmarks.

### Strengths
* The drop-in property of the approach makes it generically applicable.
* The studied topic is important for the safe use of deep learning technologies.

### Weaknesses
* There exist abundant prior work on uncertainty calibration in deep neural nets. However, the paper does not provide a comparison against the state of the art in the field. The authors can find a sizeable list of alternative methods even in an almost half-decade long paper [1]. The paper claims to have a comparison against the old Guo et al. baseline, but it is not available anywhere in the paper.

* The paper exhibits a convoluted and unstructured presentation practise. It starts from an abstract that lacks a meaningful progression of arguments and continues with a similar introduction. For example, the second sentence says the deep neural nets are miscalibrated and logically inconsistent. These two are different problems. Which one is our focus? The third sentence says these problems are problematic in structured classification tasks. What does this mean and why uncertainty calibration is problematic particularly in structured tasks? The paper introduces the studied data sets in the methodology section and does not really introduce a concrete methodology anywhere.
* The suggested combination of techniques such as differentiable fuzzy logic and focal set reasoning have not been justified anywhere. Their value added over the alternative tracks of uncertainty calibration have not been pointed out. The related comparisons to the state of the art are also missing.
* The logical trail of the suggested solution doesn‘t follow a clear rationale. Section 4 introduces the architectural elements of a standard hierarchical classifier. It then jumps to introducing some basic elements from fuzzy logic in Section 5 and an existing application of it to probabilistic deep learning called RS-CNN. However it does not explain what this prior work is doing, which aspects of it are relevant for the problem at hand, and which limitation of it will be overcome. Then Section 6 admits to follow the ROAD-R approach without explaining or motivating it, which follows some performance scores definitions. These pieces do not really come together to make a concrete scientific hypothesis. As I point out in the questions section below, all this endeavour is also missing a clearly stated purpose.
* Section 9 doesn’t specify an experiment plan. It is not possible see the big picture from the way the results are presented. Tables 2 and 3 in the appendix give further details and the only take away I can extract from these tables is that all models in comparison perform comparably.

[1] Minderer et al., Revisiting the Calibration of Modern Neural Networks, NeurIPS, 2021

### Questions
* How generalizeable are the proposed findings across different neural architectures? The Swin transformer is a very specific architecture. Why should it be the only considered backbone architecture? Why does it have to be such central in the story line? Which property of this architecture makes it representative?
* Having read the whole paper, I am left a bit confused about the end goal of the paper. Is it to improve the calibration of the uncertainty predictions of deep learning algorithms as studied in the experiments or to improve the structural consistency of the calibration methods as claimed in the first sentence of the abstract? This is not a merely aesthetic concern. If the only measurable effect of the suggested improvement will be improved calibration scores, I am missing why we need all the complications introduced by the subjective logic concepts. Furthermore, I will also then wonder why the state of the art in post-hoc uncertainty calibration is sidestepped. I would at least expect to see a comparison against temperature scaling. If the goal is to improve explainability, physical consistency, or interpretability, where is the related experiment and the result demonstrating that the suggested approach solves the problem better than what is known?

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a new neuro-symbolic architecture for hierarchical classification tasks. This architecture combinesa pre-trained swin transformers with a rather complex combination of belief functions (taking inspiration from a random set neural networks) and fuzzy logic (taking inspiration from other fuzzy-logic-based NeSy approaches). The aim is that of obtaining calibrated (low ECE) predictions that satisfy the hirarchical constraints with high probability. Experiments are carried out on two datasets and against two competitors (MultiPlexNet and RS-NN).

### Strengths
**Originality**: This is the first time I see NeSy, fuzzy logic and belief functions all combined in the same package. While the different pieces already exist, their combination is novel. No complaints on my end.

**Quality**: The overall architecture is generally sensible.

**Signfiicance**: Combining calibration and rule satisfaction is a good idea.

### Weaknesses
**Clarity**: The structure of the paper is a bit odd and many important details are not explained in an intuitive manner.

- For instance, the datasets used for evaluation are introduced in Section 3.1, before the method and far away from the experiments; it would be best to move the description to the experiments.

- I found sections 5-8 unnecessarily complicated. The authors assume the reader is familiar with belief functions, focal sets and other hyper-specialized concepts (like the architectures of RS-NN and ROAD-R). This is not necessarily the case. Equations are provided without any intuition as to what they are supposed to do. It *is* possible to make out what the authors mean, but the text doesn't make it easy. I strongly recommend the authors to provide clear intuitions for each and every equation.  Adding a figure depicting the inteded information flow in the model would also help.

Moreover, standard quantities (like the definition of Gaussian distribution, which appears twice) can be removed.

Overall, clarity is impaired by these issues and the paper as a whole feels unpolished.  It is also shorter than 9 pages (although this is just a symptom, not a problem by itself).

**Quality**: The experiments are not convincing, for several reasons.

 - They only consider two datasets. The original works by Giunchiglia (cited by the authors as an inspiration -- Coherent hierarchical multi-label classification networks; NeurIPS and its journal version) provide a **twenty** already implemented hierarchical classification tasks that could be used for evaluation. It's not clear why the authors focus on just two.

 - The choice of competitors is not ideal.  Giunchiglia's own approach is not compared against.  More recent follow-ups, such as semantic probabilistic layers [1], are not compared against.  In a nutshell, the experiments do not consider the state-of-the-art in NeSy hierarchical classification.

 - The authors also neglect NeSy approaches specifically designed for calibration, such as BEARS [2] and NeSy diffusion [3].  (Admittedly, the last one might be *too* recent, feel free to disregard it if so; BEARS, however, is not.)

- The choice of evaluation metrics is also not well motivated. Why top-1 accuracy? Why not using the same metrics used by Giunchiglia in their work and follow-ups?

Given the above, it is difficult to gauge the relative effectiveness and generality of the proposed approach.  This limitation is by itself is sufficient to make me lean toward rejection.

**Significance**: very difficult to assess, given how limited the experiments are.

[1] Ahmed et al., Semantic probabilistic layers for neuro-symbolic learning, NeurIPS 2022.
[2] Marconato et al., BEARS Make Neuro-Symbolic Models Aware of their Reasoning Shortcuts. NeurIPS 2024.
[3] van Krieken et al., Neurosymbolic Diffusion Models. arXiv 2025.

### Questions
Feel free to comment on any of the weaknesses I pointed out.

### Soundness
1

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
3

### Summary
This paper proposes a neuro-symbolic framework that combines Swin Transformers with focal set reasoning and differentiable fuzzy logics for hierarchical image classification. The approach aims to improve calibration and logical consistency while maintaining competitive accuracy on CIFAR-100 and iNaturalist datasets.

### Strengths
- This paper combines epistemic uncertainty modeling (via focal sets and Dempster-Shafer theory), fuzzy logic (t-norms), and modern vision transformers.
- The primary strength is a new integration of two distinct fields: epistemic uncertainty and neuro-symbolic reasoning. The paper makes a strong case that most prior work addresses either logical consistency or uncertainty, but rarely both in a unified manner.

### Weaknesses
- The presentation of the results is difficult to follow. The authors should consider consolidating these findings into a summary table or figure to improve readability.
- The contributions of this paper are focused entirely on the proposed method, with no accompanying theoretical analysis or guarantees.
- Only two datasets tested, both with relatively shallow hierarchies (2 levels). No comparison with recent strong baselines (e.g., hierarchical softmax, conditional probability approaches).

### Questions
- How sensitive is the model's performance to the pre-computed focal sets? How would performance change with different clustering algorithms or a different number of focal sets?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes unifying uncertainty estimation with an epistemic approach that ensures logical consistency for hierarchical image classification with pre-trained Swin transformers. They propose a two-head architecture (fine and coarse head). The epistemic component follows the strategy of RS-CNN with focal sets induced in the latent space. The logical consistency is integrated into the learning process through a belief-based, logically constrained loss, ensuring that fine-level belief masses are compatible with the coarse level. An experimental validation is provided for two datasets: CIFAR-100 and INaturalist 2021. The main contribution of the paper is the principle of unifying belief theory for epistemic uncertainty estimation with logical consistency.

### Strengths
**originality**
+ The originality of the paper relies on the idea of unifying uncertainty estimation using belief theory and focal sets with semantic regularization using a logically constrained loss. The unification principle is straightforward and logical. 

**quality**
+ The authors took care to formalize the proposed approach with a set of equations to ensure a clear understanding of the proposed regularized cost function.

**clarity**
+ The motivations of the paper are clear.

**significance**
+ The contribution addresses an important topic in AI, particularly for deep learning models, concerning their robustness and uncertainty estimation. The idea of leveraging belief theory in this context is not new, but it presents an interesting avenue for study. The proposed approach also falls within the category of neuro-symbolic approaches, integrating a priori knowledge — logical constraints — into the learning pipeline (here, in the training process, with a semantic regularization term). It is also an interesting way to study with different expected gains (robustness, frugality, explainability...).

### Weaknesses
I have several concerns about the paper.

+ **Concern 1: lack of technical novelty.**

    + The paper is strongly built on the [Random-Set Neural Network (RS-NN) paper](https://arxiv.org/pdf/2307.05772) and on classical semantic regularization on the neuro-symbolic literature. The contribution mainly relies on the integration of these two aspects. Moreover, some shortcomings in the proposed approach are not enough motivated. For instance, hierarchical classification appears to be limited to a bi-level problem with fine-grained and coarse-grained classes. What about a hierarchy with different layers? Regarding the semantic regularization part of the work, it also lacks precise positioning in relation to the state of the art on semantic regularization. See, for instance, [Xu et al,. 18](https://proceedings.mlr.press/v80/xu18h/xu18h.pdf), [Ahmed et al,. 24](https://arxiv.org/abs/2405.07387), [Ledaguenel et al,. 24](https://filuta.ai/images/compai/CompAI_paper_7.pdf). 
   + It also lacks a positioning with regard to other uncertainty estimation approaches, such as conformal prediction, for instance.


+ **Concern 2: lack of clarity.**
Although the effort to formalize the approach is commendable, it seems that the formalization should be carefully reviewed and verified.
    + What is $C$ in equation 3 ? It is not defined. While it seems clear that it is, in this context, the set of classes, as in the RS-NN paper, it should be defined. 
    + Are there any implicit constraints on the mapping function $g$? For example, can the function handle multiple parents? This function and its properties should be described in much greater detail. 
   + The clustering process of the latent representations should be more detailed. What is $\mathcal{O}$ in equation 6?

+ **Concern 3: lack of an exhaustive positioning with the neurosymbolic literature**

     + Section 6.7 is too short. Important references of the NS literature are missing. See, for instance, all the works on semantic regularization mentioned before. Moreover, an important aspect that is missing is the evaluation of the gain of the NS part.

### Questions
Main questions :

+ What about the scalability aspect of the proposed approach? Indeed, the exponential complexity of using $2^{\mathbb{Y}_f}$ and  $2^{\mathbb{Y}_c}$  sets of classes is an important issue. How is this aspect managed in the proposed approach? 
+ What about more hierarchical constraints, involving more than two levels (coarse and fine)?
+ Why a hidden representation space of size 512?
+ See previous questions regarding the mapping function $g$, on the clustering process . 
+ What is the impact of the chosen pre-trained backbone on this clustering process? 
+ What is the impact of the choices of the membership function and the chosen T-norm?

### Soundness
2

### Presentation
1

### Contribution
2
