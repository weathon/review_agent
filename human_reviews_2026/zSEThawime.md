# Neural Concept Verifier: Scaling Prover-Verifier Games via Concept Encodings

- Decision: Reject
- Scores: 2, 6, 6, 4

## Abstract
While *Prover-Verifier Games* (PVGs) offer a promising and much needed path toward verifiability in nonlinear classification models, they have not yet been applied to complex inputs such as high-dimensional images. Conversely, *Concept Bottleneck Models* (CBMs) effectively translate such data into interpretable concepts but are limited by their reliance on low-capacity linear predictors.  
In this work, we push towards real-world verifiability by combining the strengths of both approaches. We introduce *Neural Concept Verifier (NCV)*, a unified framework combining PVGs for formal verifiability with concept encodings to handle complex, high-dimensional inputs in an interpretable way. NCV achieves this by utilizing recent minimally supervised concept discovery models to extract structured concept encodings from raw inputs. A *prover* then selects a subset of these encodings, which a *verifier*, implemented as a nonlinear predictor, uses exclusively for decision-making.  
Our evaluations show that NCV outperforms CBM and pixel-based PVG classifier baselines on high-dimensional, logically complex datasets and also helps mitigate shortcut behavior. Overall, we demonstrate NCV as a promising step toward performative, verifiable AI.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
Concept-based models (CBMs) combine neural concept extractors with a classifier that only takes concepts as input. The latter classifier is often a simple linear output layer, limiting expressivity and reducing the accuracy compared to end-to-end neural network models. This paper proposes to replace that linear layer with a "Merlin-Arthur classifier (MAC)", a recent GAN-like formalism where a neural network model selects certain concepts, and another neural network model uses that subset of features to classify the output. This MAC is claimed to retain interpretability.

### Strengths
- Novel approach, I did not know about these MACs. 
- Good amount of clearly described experiments with evidence for the method.
- Writing is overall clear and easy to follow

### Weaknesses
- Fails to discuss and compare to other works that tackle the same limitation of CBMs
    - Because of this existing body of literature, somewhat incremental
- It is unclear what the theoretical benefits of interpretability and verifiability of MACs are
- A few vital details for reproducability are missing
- No code provided

### Questions
Critical:
- The authors claim "the classifier component of CBMs has received little attention" (107). However, this has already been studied since 2022 [1], including particularly the accuracy-interpretability tradeoff. Several papers followed up on this, using various classifiers that have different tradeoffs. Here are at least a few I'm aware of [2-4]. None of these are mentioned in the manuscript. 
    - I believe there should at least be a baseline of some of these papers (at least [1, 3], I'd say)
    - Furthermore, the related work should discuss the benefits of NCV compared to these existing approaches
- The abstract, introduction and conclusion mention the '(formal) verifiability' of NCV by using a 'verifiable classifier'. However, the manuscript contains no clear theoretical claims as to what exactly can be verified. I guess these are inherited by using MACs, but it should be clearly stated in the paper with the relevant assumptions needed. 
- The above criticism also holds for the claim that "NCV emphasizes faithfulness and interpretability" (253). Why are explanations faithful? And why are they interpretable? As far as I understand, there is nothing preventing the Arthur model from completely misunderstanding the concepts as long as it gets high classification accuracy. 
- The loss is not differentiable with respect to Merlin and Morgana, as it involves discrete subset selection. I could not find how NCV estimates gradients for these models. 

If the above comments are resolved, I would be happy to increase my score. 

Other comments and questions
- The paper mentions a 'rejection class' to use for adversarial inputs. However, it is not clear why it would ever predict it, as it is not the case the model has to predict 'reject' for Morgana. 
- Q3 about 'more detailed explanations' was surprising. NCV does not provide more detailed explanations to me, rather more abstract and higher-level than pixel level. Further, the claim that 'NCV ... pinpoints the precise attributes relevant for a class decision" (407-408) seems unproven from 5 qualitative examples, and it is unclear if these explanations are causally reliable [5] / causally opaque [4]. 

1. Espinosa Zarlenga, M., Barbiero, P., Ciravegna, G., Marra, G., Giannini, F., Diligenti, M., ... & Jamnik, M. (2022). Concept embedding models: Beyond the accuracy-explainability trade-off. Advances in neural information processing systems, 35, 21400-21413.
2. Debot, D., Barbiero, P., Giannini, F., Ciravegna, G., Diligenti, M., & Marra, G. (2024). Interpretable concept-based memory reasoning. Advances in Neural Information Processing Systems, 37, 19254-19287.
3. Barbiero, P., Ciravegna, G., Giannini, F., Zarlenga, M. E., Magister, L. C., Tonda, A., ... & Marra, G. (2023, July). Interpretable neural-symbolic concept reasoning. In International Conference on Machine Learning (pp. 1801-1825). PMLR.
4. Dominici, G., Barbiero, P., Zarlenga, M. E., Termine, A., Gjoreski, M., Marra, G., & Langheinrich, M. (2024). Causal concept graph models: Beyond causal opacity in deep learning. ICLR.
5. De Felice, G., Flores, A. C., De Santis, F., Santini, S., Schneider, J., Barbiero, P., & Termine, A. (2025). Causally reliable concept bottleneck models. arXiv preprint arXiv:2503.04363.

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
The paper introduces Neural Concept Verifier (NCV), a method that combines Concept Bottleneck Models (CBMs) with Prover-Verifier Games (PVGs) to improve robustness against shortcut learning while maintaining interpretability. NCV uses a concept encoder to map inputs to a concept space and employs a prover-verifier framework to ensure that predictions are based on meaningful concepts rather than spurious correlations. The method is evaluated on several image benchmarks, demonstrating improved robustness and interpretability compared to some existing approaches.

### Strengths
1. Framing prediction as a PVG on a concept bottleneck is an interesting approach to the problem of shortcut learning. 
2. PVGs offer a different and more stable spin on adversarial training by shifting the adversarial intervention to the concept space instead of the input space. 
3. The proposed method provides good semantic explanations without compromising predictive performance, which is a key challenge in interpretable ML. 
4. The paper is easy to follow.

### Weaknesses
1. The claim that CBMs are "limited by their reliance on low-capacity linear predictors" is not quite fair; CBMs can easily be combined with non-linear predictors, so this is not really a serious limitation that this paper resolves. 
2. The first 3 benchmarks in Table 1 are missing a key baseline, namely, CBM on the CLIP-Sim feature space with a nonlinear probe. This is to test whether the PVG approach has a distinct performance edge over simply applying CBMs on a rich concept vocabulary with nonlinear probes. 
3. Table 2 is missing the Pixel-MAC baseline, which is important to see how NCV compares to other PVG methods on shortcut learning. 
4. It is unclear how significant the robustness gains are since the paper assumes a safe (non-adversarial) concept encoder, which ignores an important risk/attack vector of shortcut learning where the concept encoder itself may learn spurious correlations.
5. (Minor) The last sentence of the Table 1 caption, "NCV consistently matches or outperforms baselines in completeness," is not totally accurate since the CBM baseline outperforms NCV on ImageNet-1k in terms of completeness.

### Questions
1. Can you provide an ablation study on the concept mask size? You report the results for 32 concepts, but it would be interesting to see the relationship between performance and the concept count. 
2. Why do you think Pixel-MAC performs poorly on CIFAR-100, ImageNet-1k, and COCOLogic compared to CLEVR-Hans? 
3. How was the CLIP-Sim vocabulary chosen? Was it selected based on relevance to the datasets, or was it a generic set of concepts? 
4. Can you report the validation–test gap for Pixel-MAC? 
5. How does the performance change if you only optimize Merlin's loss (i.e., $\gamma=0$)? Can you report the performance and loss curves for different values of $\gamma$? 
6. Can you evaluate your method against input-space adversarial attacks (e.g., FGSM) to see if the robustness gains in concept space translate to input space robustness? 
7. What are some real-world image-based settings where adversarial provers would be relevant? Real-world settings for pixel-level attacks are fairly established, but I'm not aware of ones where a concept selector may be compromised in the image domain.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces **Neural Concept Verifier (NCV)**, a novel framework for interpretable image classification. The main contribution of this work is the integration of concept extractors into Prover-Verifier Game (PVG) frameworks achieving a joint training of three agents in the concept embedding space. Evaluations were done across 5 datasets, where different instantiations of NCV were tested (e.g. extractor=NCB, pgvs=Set Transformers). Results support the claimed strengths of the NCV framework, where SOTA accuracy is achieved on 4 out of 5 datasets with an additional certificate of robustness to shortcuts (low validation-test gaps). Output explanations are compared to pixel-MAC (an instantiation of PVG directly applied to image pixel space), showing human readable (text) concepts identified in the image that supports the predicted class.

In summary, this work reduces the interpretability-accuracy gap by demonstrating SOTA performance in image classification while outputting interpretable explanations.

### Strengths
S1: this work is well-motivated by the gap of sacrificing accuracy for interpretability or vice versa in current literature, and the results generally supported all the claims stated.

S2: the framework is designed to be generalizable -- it allows integration of different concept extractors and verifiers, showing the potential for future studies on this framework.

S3: scalability is studied on real-world image datasets such as ImageNet-1k.

S4: robustness for shortcuts is studied and discussed, where results support the claim of NCV being robust to shortcuts.

### Weaknesses
W1: sometimes it is not clear to the reader what is the difference between NCV and MAC as in section 3 it seems like most of NCV is just adapting MAC. The authors should clarify and emphasize on their innovations beyond MAC (for example, there are a couple places where the authors states ".... shifting the PVG to concept encodings ..." (line 343), it can be made more clear if the authors mention the original set up of PVG such as "... shifting the PVG from xxxxxx to concept encoding ..." to highlight the contribution.

W2: the reader assumes that the shown concepts/explanations were textual because the extractor's output was textual. This indeed shows interpretability, however, it is not clear to the reader whether the quality of these concepts were controlled. For example, in Figure 3 for the coffee-mug, "extend" and "support" were two concepts output before "cappuchino", which seems to be irrelevant in the image, defeating the purpose of being interpretable. It might help if sub-regions in the image could be linked to each concept, achieving a better explanation and especially trustworthiness.

W3:  it was not discussed how much NCV might cost more than the baselines, for example, training time, hardware requirements, set-ups needed for replication of this work. 

W4: it is not clear to the reader how "Soundness" was evaluated in Table 1, is it the same as the Robustness as in Table 2? What was the metric used for soundness?

W5: on line 377 it was claimed that NCV improves over linear CBMs in both accuracy and robustness, however, in table 1 CBM was not even tested on robustness (results are shown as n/a).

### Questions
It will be very helpful if the authors could answer the following questions:

Q1: In figure 2, when the concepts were output as "xxxx and yyyy and zzzz", are the "and" used for the presence of the concept (totally relying on the extractor used)? What if sometime it is the absence of a certain feature that drives the prediction? How would NCV output explanations in those cases?

Q2: on line 167, is C (the number of discovered concepts) a fixed or learnable parameter?

Q3: How are provers and verifiers selected for different datasets/tasks? Suppose we would like to apply NCV to a new image dataset, is there any property or guideline on how one should select the model architectures for the verifier and prover combination? Any pilot studies or ablations done for different combination of prover+verifier for the same dataset? Ultimately, how much does the selection of the combination affect the performance of NCV?

Q4: see W3

Q5: see W4

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
4

### Summary
This paper introduces the Neural Concept Verifier (NCV), a framework that integrates Prover–Verifier Games (PVGs) with Concept Bottleneck Models (CBMs) to achieve interpretable and verifiable classification on high-dimensional data such as images. By shifting the prover–verifier interaction from pixel space to concept encodings, NCV aims to combine the formal verifiability of PVGs with the interpretability of concept-based reasoning. The authors evaluate NCV on synthetic and real-world datasets (CLEVR-Hans, CIFAR-100, ImageNet-1k, and COCOLogic) and report improvements in completeness, soundness, and robustness to shortcut learning compared to standard CBMs and pixel-level PVG models.

### Strengths
The central idea, combining PVGs and CBMs, is intriguing

The focus on verifiability is timely and relevant for trustworthy AI.

The experimental setup is diverse, covering both synthetic and real-world datasets.

The exposition is generally clear and well organized, though sometimes too high-level.

### Weaknesses
**Lack of comparison with relevant recent models** 
The paper compares only against plain CBMs and pixel-based PVGs. However, there has been a recent surge in models that incorporate interpretable yet nonlinear mappings—often grounded in logic or symbolic reasoning (e.g., [1], [2]). In particular, Debot et al. maintain a global logic-based task decoder that allows the use of theorem provers to provide formal proofs of desired logical properties. These works appear conceptually close to NCV, yet no theoretical and/or empirical comparison is provided.

[1] Barbiero, Pietro, et al. "Interpretable neural-symbolic concept reasoning." International Conference on Machine Learning. PMLR, 2023.

[2] Debot, David, et al. "Interpretable concept-based memory reasoning." Advances in Neural Information Processing Systems 37 (2024): 19254-19287.


**Unclear guarantees and metrics for verifiability** 
The “soundness” metric is not formally defined. Since it depends on a jointly trained component, it is unclear what guarantees it offers against an external adversary. The Arthur verifier might be robust only to the attacks generated by Morgana, rather than to arbitrary ones. The underlying statistical assumptions should be explicitly stated.

**Ambiguous definitions of verifiability and robustness** 
These are central to the paper’s claims, yet their formal definitions and relationship to completeness/soundness remain unclear. A deeper conceptual and formal grounding is needed.

**Unconvincing shortcut-learning setup** 
The rationale for interpreting corruptions as shortcuts is unclear. The distinction between “clean samples” and “shortcut-free” data is not well defined. Conceptually, shortcuts arise when multiple predictive routes exist, making the human-desired one unidentifiable. This is akin to standard arguments in regularization theory. Adding corruption could instead eliminate or distort the “human” model entirely. The argumentation here is weak and conceptually confused. Moreover, adversarial setups can have a regularizing effect, which may confound the reported improvements. Comparisons with standard shortcut-mitigation strategies (e.g., regularization, reconstruction losses) would be more informative.

**Minor clarity issues** 
- Figure 3 gives the impression that an object-centric model can segment individual objects in CLEVR-Hans, which does not seem to be the case. The same explanatory style as for ImageNet-1k would be preferable.
- The description of the shortcut experiments could benefit from a better description for clarity.

### Questions
- How does NCV theoretically or empirically differ from recent logic-based interpretable models (in particular with [2])?

- What formal guarantees does the “soundness” metric provide beyond internal consistency with the Morgana prover?

- Why should data corruptions correspond to shortcuts, and how does this align with the standard definition of shortcut learning?

### Soundness
2

### Presentation
3

### Contribution
2
