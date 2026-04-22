# Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 6, 6, 8, 8, 6

## Abstract
Recent empirical studies have explored the idea of continuing to train a model at test-time for a given task, known as test-time training (TTT), and have found it to yield significant performance improvements.
However, there is limited understanding of why and when TTT is effective.
Earlier explanations mostly focused on the observation that TTT may help when applied to out-of-distribution adaptation or used with privileged data.
However, the growing scale of foundation models with most test data being in-distribution questions these explanations.
We instead posit that foundation models remain globally underparameterized, with TTT providing a mechanism for *specialization after generalization*—focusing capacity on concepts relevant to the test task.
Specifically, under the linear representation hypothesis, we propose a model in which TTT achieves a substantially smaller *in-distribution* test error than global training.
We empirically validate our model's key assumptions by training a sparse autoencoder on ImageNet, showing that semantically related data points are explained by only a few shared concepts.
Finally, we perform scaling studies across image and language tasks that confirm the practical implications of our model, identifying the regimes where specialization is most effective.
Beyond TTT, our results provide additional strong evidence in support of the linear representation hypothesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The authors analyze a model of test time training and argue that test-time training can provide benefits even in-distribution, so long as the model is capacity constrained.

### Strengths
- The paper is well-written and easy to follow
- The observation that TTT can help performance even for in-distribution inputs, and that this benefit is more pronounced in the underparameterized regime, is interesting, and to my knowledge, novel
    - The authors show how the effect of TTT may be to 'disentangle' representations in an underparameterized representation space
- The authors confirm their predictions with experiments across scales & domains: from MNIST to language models

### Weaknesses
- The authors consider TTT only in the regime where the last linear layer is retrained. While this does ensure that the representations learned during pretraining remain unchanged, it also puts a capacity constraint on the TTT model. Do the observations in the paper still hold up if we train more of the model at test time? 
    - I am particularly concerned that the phenomena in the paper may be a special case induced by the linearity of the final layer. I think even an experiment where TTT modifies the last 2 layers would go some way to mitigating this concern. 
- It is not clear to me where the under/overparameterized regimes begin and end. Could the authors note this on the figures? 
- In Figure 4, at least for the MNIST dataset, if the authors' hypothesis is correct, it should be possible to push to a regime where TTT yields no improvement. Is it feasible to extend the x-axis further, at least for the MNIST models?

### Questions
- In the theoretical framing, the relevant dimensionality seems to be the dimension of the latent space (compared to the number of "concepts" needed to solve the task). Despite this, in several places (e.g. Fig 4), the *total* number of parameters is used as a proxy for this dimensionality. Are there experiments you could run to disentangle these two different notions of dimensionality? For example, changing width/depth of MNIST models independently, do you see different behavior?
- Can you clearly delineate the under/overparameterized regimes on Figs 4,5? It is not clear to me how much training data was used in each setting.
- Is there a way to show that during TTT, the "effective" dataset size (during TTT we allow the model to perform poorly on data not in the current test set, so the effective dataset is small) becomes small enough that the model is again ~overparameterized? Can you connect this to double descent, at least in the MNIST setting? 
    - I am speaking loosely here, but the idea is that doing TTT in-distribution amounts roughly to retraining the model on a smaller dataset, which we know from the double descent literature has the potential to *improve* performance.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper attempts to answer why and when does test-time training (TTT) help foundation models even when test data are in-distribution? It assumes that foundation models are under-parameterized, and that TTT provides "specialization after generalization". Behind this is the Linear Representation Hypothesis (LRH): high-level "concepts" are sparse and linearly represented but superimposed onto a lower-dimensional feature space.

The authors reach 3 key observations: (O1) neighborhoods in learned features preserve the geometry from the concept space; (O2) each test neighborhood only depends on a few active concepts; (O3) standard TTT implicitly finds sparse solutions in concept space. Usinng sparse autoencoders and scaling studies on MNIST, ImageNet (with CLIP features), and The Pile (with LoRA), they show that TTT improves accuracy especially when models are small and under-parameterized, and that the benefit shrinks as models scale up in terms of number of parameters. Overall, the paper provides evidence that TTT’s success comes from reallocating model capacity locally.

### Strengths
Comprehensive and helpful literature review. Text is well-written and structured. Figures are clear and support main claims.

The three observations (O1--O3) are backed up with both experiments and analysis.

The math connects with the experimental results and explains why global training can fail when features are mixed.

Understanding when TTT helps (even with no distribution shift) is relevant for practical deployment of large models. The paper offers practical guidelines for scaling, such as how to choose neighborhood size and when TTT is likely to be worth using.

### Weaknesses
Several results depend on strong ideas like sparse concept spaces and preserved neighborhood geometry. It would help to test these assumptions on different architectures or layers to show they hold more generally.

The LoRA test-time training uses only one step and on a small subset (1%), 50 neighbors. Running more steps, larger splits, or other LLMs could show whether the findings scale.

The tests on images (LeNet-5 and pre-trained CLIP ViT) only update the last layer. Adding results with full end-to-end fine-tuning would make the findings more convincing.

Although the paper does a good job at listing all its main assumptions, it mostly deals with them by giving arguments for why those assumptions might be reasonable. But the paper is lacking on discussing what could happen in scenarios where they do not hold. For example, TTT assumes neighborhoods correspond to a single uniform local concept. The authors should discuss what happens when that assumption breaks (e.g. due to label noise, or poor neighborhood size selection), since specialization in that case may lead to worse generalization.

### Questions
In the LoRA setup, the paper uses one gradient step. How was this chosen? Were multiple step sizes or LoRA ranks tested? Would they improve results?

In Appendix E.4, Table 4, different values are reported for the number of neighbors used for TTT on MNIST, depending on the model size. How were these chosen? In contrast, why was a fixed value of 100 used for ImageNet and of 50 for language modeling?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
1

### Summary
This paper investigates test-time training (TTT) and proposes that its effectiveness arises from enabling specialization after generalization, where models locally adapt to test data. Using the linear representation hypothesis, the authors model how TTT reweights a small subset of relevant concepts in an underparameterized feature space, achieving lower in-distribution error than global training. Empirical and theoretical results across vision and language tasks show that TTT offers the greatest gains for underparameterized models, with benefits diminishing as model capacity increases.

### Strengths
1. The paper introduces a clear and formal connection between TTT and the linear representation hypothesis (LRH), providing an interpretable mathematical model for understanding why and when TTT improves performance.
2. The use of SAEs to approximate the latent “concept space” offers a convincing and interpretable lens to study TTT’s behavior, even for readers (like myself) who are not deeply specialized in TTT.
3. Experiments on both image (ImageNet, MNIST) and language (Pile) domains systematically support the theoretical claims, demonstrating consistent trends across scales and tasks.

### Weaknesses
As someone not specialized in test-time training, I found the exposition somewhat difficult to follow—the paper would benefit from a clearer introductory section that outlines core TTT mechanisms and variants before diving into the theoretical framing.

### Questions
Could the author elaborate on the definition of test-time training?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper studies the mechanisms of TTT in Foundation Models and its effectiveness in different contexts (the model capacity or dataset complexity). The work asks an important question of the TTT's effectiveness performed on the already seen data samples, which lets us focus on the process itself and not on the data. The claims made by the authors are clearly stated and supported by a thorough experimental section spanning experiments from image classification to NLP.

### Strengths
1. The work focuses on the extremely important topic, as the TTT plays an important role in the current cutting-edge research after showing its effectiveness in different domains; thus, I value the significance of the results highly. 
2. The idea to isolate the effect of TTT from the data (i.e. to study the effectiveness of TTT on already seen data) makes the analysis much more convincing to me. Still, I have to point out that I'm only vaguely familiar with this field, so my novelty judgment can be off. 
3. Both the presentation and the experimental design are thoughtful and help the reader to understand the main claims made by the authors, making it a solid piece of work with carefully balanced claims.

### Weaknesses
I did not find any important weaknesses in the paper. I have some questions which I will ask in the next section.

### Questions
1. Can we conclude that current LLMs, given their performance gains after TTT, are underparameterized models relative to their datasets? If so, why do the authors think that scaling the LLMs to even greater capacities ceased to be the main theme in LLM training, and we started to observe the diminished returns from scaling? 


2. How important is the metric used for generating the neighbourhood? The authors studied L2 distance, which was equivalent to cosine similarity, but there are multiple different ways to define the neighbourhood. Can we expect this metric to be, on average, the best approach, or would this choice be important when applying the approach to different datasets, modalities or architectures?

3. Line 239-241: The Authors claim that the sparse TTT classifier is functionally equivalent based on 89% match (except for pathological cases). Could the authors elaborate on what "pathological" means? 11% seems to be relatively high and I don't understand where should I expect these two would deviate.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors conduct a comprehensive investigation into the potential of test-time training (TTT) to reduce in-distribution test error by leveraging only previously seen data. While TTT has traditionally been employed to address distribution shifts, this work explores its applicability to in-distribution scenarios. The authors first demonstrate that TTT encourages the model to converge to sparse solutions in the concept space. Building on this insight, they empirically show that TTT can effectively reduce in-distribution error rates on MNIST, ImageNet, and language modeling.

### Strengths
- The idea of leveraging test-time training (TTT) to reduce in-distribution error is both novel and well-motivated, particularly given that most prior work on TTT has focused on addressing distribution shifts. The authors indeed demonstrate that TTT effectively reduces in-distribution test error compared to standard global training. 
- The analysis of the model’s representation space, grounded in the linear representation hypothesis, provides a solid conceptual foundation for understanding how models learn and encode semantic concepts.
- The paper presents a rigorous theoretical analysis of the in-distribution test error behavior under TTT, which strengthens the overall contribution and supports the empirical findings.

### Weaknesses
- I believe the term “foundation model” requires clearer definition in the context of this paper. As shown in Figure 4, the effectiveness of test-time training (TTT) in reducing in-distribution error appears to diminish as model capacity increases. This observation raises concerns about the applicability of TTT to large-scale foundation models, which are typically defined as models with significantly larger parameter counts (e.g., exceeding 32 billion parameters). This concern is also acknowledged in Takeaway 1. However, in the introduction, the authors state that foundation models are underparameterized, which seems to contradict the commonly accepted understanding of the term. This inconsistency may lead to confusion for readers, and I believe a more precise and consistent definition of foundation model would help clarify the paper’s contributions and claims.
- Although not a major concern, the experiment using MNIST with LeNet-5 appears somewhat outdated. While it is understandable that the authors aimed to demonstrate their method on a simple dataset and model, the use of a more recent dataset and architecture would have strengthened the empirical evaluation, particularly in light of the paper’s claims regarding foundation models.

### Questions
Please refer to my weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3
