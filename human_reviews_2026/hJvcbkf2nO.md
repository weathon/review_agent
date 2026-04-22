# Model Stitching by Invariance-aware Functional Latent Alignment

- Avg Score: 3.50
- Decision: Reject
- Scores: 6, 4, 2, 2

## Abstract
In deep learning, functional similarity evaluation quantifies the extent to which independently trained models learn similar input-output relationships. A related concept, representation compatibility, is investigated via model stitching, where an affine transformation aligns two models to solve a task. However, recent studies highlight a critical limitation: models trained on different information cues can still produce compatible representations, making them appear functionally similar \cite{smithfunctional}. To address this, we pose two requirements for similarity under model stitching, probing both forward and backward compatibility. To realize this, we introduce invariance-aware Functional Latent Alignment (I-FuLA), a novel model stitching setting. Experiments across convolutional and transformer architectures demonstrate that invariance-aware stitching settings provide a more meaningful measure of functional similarity, with the combination of invariance-aware stitching and FuLA (i.e., I-FuLA) emerging as the optimal setting for convolution-based models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper considers a few recently discovered issues with representation similarity characterizations, that indicate that they are either overly permissive (like stitching with task loss matching, where the stitching layer can get "too creative" in achieving a good task loss) and too penalizing (eg stitching with direct matching, where there is too much focus on fitting a single layer as closely as possible, without regard of the dynamics of the network as a whole).

The paper proposes two techniques. One is a new loss for stitching called functional latent alignment (fula) that involves matching all the layers after the stitching layer, and the other is the utilisation of identically represented inputs as a means to examine whether the inputs that are represented similarly indeed are processed similarly.

The paper then presents an empirical evaluation of the method and compares it with known baselines, showing that the indentically represented inputs are indeed a good tool for separating real difference from "cheating" stitchings.

### Strengths
The paper introduces very interesting ideas about fixing the current stitching approaches in order to get a cleaner insight into representation similarity. The main idea seems to be that one needs to look at both the representations in each layer as well as the expected invariances while propagating the representations.

The empirical evaluation is interesting and indeed supports the claims that the proposed techniques add a new, useful perspective.

### Weaknesses
The main problem is with the presentation. The paper is very hard to read, even for someone who is familiar with the area. The paper is extremely dense; there is a lot of content packed in a limited space, and so things are not sufficiently motivated, explained, and discussed. The plots are extremely small, and even in color, they are difficult to read. It requires a lot of concentration to understand what the plots show and how the experiments were conducted. At the same time, Figures 1 and 2 take up a lot of space, while I found them a lot more confusing than helpful. Even after understanding the text, I still had a hard time making sense of these plots. I personally don't think they are necessary, at least in the main text, and then you could have larger plots and more words to explain what is going on.

As for the method, it is evident that req B does the heavy lifting, while FULA is not that different from SLM. So I was not entirely convinced that FULA is even necessary. I think the idea of input invariance is the key here. However, I found the motivation of req B less clear, I think it would need some more support and explanation. Even sec 2.3.1 is quite confusing because it is not clear how we take care of req B exactly. (Later becomes somewhat clearer, but essentially just from the way you construct the plots.)

### Questions
Are there any cases when FULA is clearly necessary and "better" (in the sense of some sanity checks) than SLM?

You promise at some point that some sanity checks are being used (lacking any formal "oracle), which is fine, but then you do not seem to state your sanity checks clearly and up front. What are these?

### Soundness
3

### Presentation
2

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
This paper investigates the problem of functional similarity in deep neural networks using the model stitching paradigm. The authors argue that existing stitching methods, which primarily focus on "forward compatibility" (i.e., maintaining task performance), can be misleading. They show that these methods often find high similarity even between models trained on different "information cues" (e.g., color vs. texture). To address this, the paper introduces two requirements for a more meaningful similarity measure: (1) latent-level forward compatibility, ensuring internal representations transition similarly after stitching, and (2) "backward compatibility", ensuring inputs that are invariant to the first model are also treated similarly by the second.

The paper proposes a new stitching objective, Functional Latent Alignment (FuLA), to enforce forward compatibility, and an "invariance-aware" training setup using Identically Represented Inputs (IRIs) to probe for backward compatibility.

### Strengths
- The paper addresses the critical problem of understanding and quantifying the similarity between neural network representations, which is fundamental to interpretability and model understanding.

- The paper's primary conceptual contribution is the introduction of "backward compatibility" as a necessary condition for meaningful similarity.

- The authors conduct a comprehensive set of experiments across multiple architectures (ResNet, VGG, ViT) and under various conditions, including different data cues and model robustness settings.

### Weaknesses
1. The paper is difficult to follow. The writing is often dense, and key concepts like "information cues" are used without a precise definition. Most importantly, Figures 4, 5 and 6 are not well-explained in the caption, are hard to interpret and it is extremely hard to match the trace to the correct item in the legend. 

2. The interpretation of some results is questionable. For example, in the cross-data stitching experiment (CIFAR-RGB vs. CIFAR-grayscale), the sharp performance drop for I-FuLA is presented as a success. However, one could argue that since the underlying image content is the same, with different augmentations, a good similarity metric should yield high similarity. The paper does not defend *why* this sharp drop is a desirable property.


3. The paper's core premise, that models trained on different "information cues" should be considered functionally dissimilar, is not sufficiently motivated. This stance appears to contradict a growing body of work suggesting that models can and should learn compatible or geometrically aligned representations if the underlying data semantics are the same (e.g., the Platonic Hypothesis). The paper fails to adequately position itself against this literature, for example the works on relative representations are cited but sligthly misinterpreted (Moschella et al., 2022, Cannistraci et al., 2023): they have already been used as invariance-aware similarity measure between representations  (e.g., Section 4.1 in Moschella et al) and not only for model stitching.


4. The analysis is missing crucial ablations. The expressivity of the stitching transformation S is fixed to a linear layer. However, the capacity of this transformation is a critical factor that could heavily influence the stitching outcome. An analysis with different capacities (e.g., identity, a small MLP) is needed to disentangle the effects of the stitching objective from the effects of the transformation's capacity.

### Questions
- Could the authors clarify their position with respect to the emerging similarity literature? These works suggest that compatible representations should emerge from data with shared semantics, even if the inputs differ (e.g., images and captions). Why should models trained on grayscale vs. RGB images be considered functionally dissimilar?

- Regarding the cross-data stitching experiment (Fig. 4, "Cross-data"), the authors present the sharp decline in performance for I-FuLA as a positive outcome. Further elaboration is needed on why this is a desirable result. An alternative viewpoint is that the models should be able to find common ground, as the semantic content is largely preserved between colored and grayscale images.

- How do the results and conclusions change when varying the expressivity of the stitching transformation S (e.g., using a multi-layer perceptron instead of a single linear layer)? It seems possible that a more expressive stitch layer could overcome the dissimilarities that I-FuLA is designed to detect, which would challenge the paper's conclusions.
 
- The paper would benefit from a more high-level, intuitive explanation of the core concepts before diving into the formal notation. This would make the work more accessible.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces I-FuLA, a new model stitching method to better measure functional similarity. It combines a novel objective, Functional Latent Alignment (FuLA), with an "invariance-aware" setting that learns the alignment on inputs with identical internal representations (IRIs). Experiments show this method provides a more meaningful similarity score, as it can distinguish between models trained on different visual cues and avoids exploiting spurious shortcuts. This leads the authors to conclude that robust and non-robust networks are less functionally similar than previously believed.

### Strengths
* **S1**: The paper tackle an interesting problem of stitching different neural networks and evaluating latent similarities.
* **S2**: The paper introduce the backward compatibility, as a new and useful rule for measuring latent similarity.

### Weaknesses
* **W1. Clarity and Presentation**: The major weakness is that the paper is dense and can be difficult to follow. It doesn't follow a clear and linear story. Additional, the core concepts of "forward" and "backward" compatibility, while central to the paper, are not introduced with sufficient clarity early on. The notation, though systematic, adds to the cognitive load. The figures, particularly the "stitching plots," are small and contain a lot of information, making them hard to decipher without extensive cross-referencing with the text. A more guided walkthrough of one of the plots in the main text would have been beneficial.

* **W2. Limited Experimental Scope**: The experimental validation is conducted on relatively small-scale datasets (CIFAR-10 and a 10-class subset of ImageNet) and primarily with one architecture (ResNet-18). While VGG-16 and ViT-Tiny are included in the appendix, the main claims are built on the ResNet-18 results. The findings would be much more compelling if demonstrated on larger-scale benchmarks (e.g., the full ImageNet-1k) and with a more diverse set of modern architectures, especially larger Transformers.

* **W3. Novelty and Contribution Statement**: The paper's primary novelty lies in formalizing the "backward compatibility" requirement and using IRIs to test it. However, this could be stated more directly in the introduction and contributions list. The introduction of I-FuLA, while new, appears to be a secondary contribution, as the "invariance-aware" setting is what drives most of the significant results. The paper could be improved by more clearly delineating the impact of each of these two contributions.

* **W4. Comparison to Other Metrics and Related Work**: The paper does not compare its similarity findings to other representation similarity metrics like Centered Kernel Alignment (CKA) [1] or to other more recent works such as [2]. Such a comparison would help contextualize their results and clarify what unique insights the notion of "functional similarity" provides over geometric or statistical similarity of representations. Additional, the authors could consider including in the related work section the following model-stitching works [4,5,6].

* **W5. Reproducibility**: Providing the code would be essential for the community to verify the results and build upon this work.

* **W6. Subjectivity of "Meaningful Similarity**: A core claim of the paper is that it provides a more "meaningful" measure of similarity. However, "meaningful" is never formally defined and is instead based on intuitive sanity checks. While the experiments are convincing, the paper would be stronger if it could connect its measure to a more concrete, objective property, or discuss the philosophical underpinnings of what makes a similarity measure meaningful.


---
[1] Kornblith, Simon, et al. "Similarity of neural network representations revisited." International conference on machine learning. PMlR, 2019.

[2] Fumero, Marco, et al. "Latent functional maps." ICML 2024 Workshop on Geometry-grounded Representation Learning and Generative Modeling. 2024.

[4] Maiorca, Valentino, et al. "Latent space translation via semantic alignment." Advances in Neural Information Processing Systems 36 (2023): 55394-55414.

[5] Cannistraci, Irene, et al. "Bootstrapping parallel anchors for relative representations." ICLR Tiny Paper (2023).

[6] Lähner, Zorah, and Michael Moeller. "On the direct alignment of latent spaces." Proceedings of UniReps: the First Workshop on Unifying Representations in Neural Models. PMLR, 2024.

### Questions
* **Q1. Generality for Transformers**: The results indicate that for ViT-Tiny, the proposed I-FuLA is not the optimal setting, and I-SLM is preferred. Does this imply that the definition of "meaningful functional similarity" is architecture-dependent, and that different model families may require different criteria?
* **Q2. Scalability to Larger Datasets**: How would the authors expect these findings to translate to more complex, large-scale benchmarks like the full ImageNet-1k dataset? The generation of the DIRIs dataset seems computationally intensive; is this approach feasible at that scale?
* **Q3. Scalability to Larger Models**: How would the authors expect these findings to translate to larger networks, such as larger Vision Transformers (e.g., ViT-Small/Base/Large) or models like DINO?
* **Q4. Framing of Similarity as a Limitation**: In the abstract, the fact that models trained on different information cues can produce compatible representations is framed as a "critical limitation." Could the authors elaborate on why this is a limitation, rather than an interesting property of neural networks (e.g., demonstrating that different paths can lead to functionally similar solutions)?
* **Q5. On the Role of the Stitching Layer's Capacity**: The experiments use a 1x1 convolutional layer for stitching. Could the results be sensitive to the capacity of this transformation layer? Is it possible that models appear dissimilar simply because a simple affine transformation is insufficient to align their representations, and a more powerful non-linear "stitching function" might reveal deeper similarities?

### Soundness
1

### Presentation
1

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
This work proposes a new approach to model stitching to measure functional alignment to two different models. The proposed approach directly matches the representations for every layer after the stitching point. The authors perform a wide range of experiments that show this stitching approach is more sensitive that alternative methods for distinguishing models. The authors also look at the effect of stitching on a different dataset with slightly perturbed representations at the stitching layer.

### Strengths
The authors present an interesting conceptual argument for this new model stitching approach, motivated by previously identified weaknesses of existing stitching methods. They then go on to perform a series of experiments based on tests conducted in prior work.

### Weaknesses
While conceptually interesting, the results appear to show that direct matching (DM) gives effectively the same interpretation as the proposed FuLA method. This makes sense since FuLA would only differ from DM when the match is poor. It would seem that DM would then be the preferable method for its simplicity. FuLA also requires the two networks being compared to have the same architecture, which DM does not require.

Moreover, the results of the paper are poorly presented. Broadly speaking, the figures are confusing to interpret due to poor labeling, minimal captions, and the size of the text, which makes most of the results almost impossible to read on paper. For example, the titles of the plots in Fig. 4 are never defined or referenced elsewhere in the paper or in the caption. Fig. 6 is even more confusing, where the x-axis label seems to be important but is never explained.

### Questions
1. For the Identically Represented Inputs (IRI) datasets, why is it necessary to generate a completely new dataset if your goal is simply to perturb the output of the front model at the stitching layer? Why not avoid the whole optimization problem and just directly perturb/add noise to the representation at the stitching layer?
2. On line 328, why is the conclusion that the forward compatibility notion doesn't differentiate among different models when it clearly shows different behavior (the referenced "dip") in the plot? It seems to clearly differentiate it in this case.
3. Why use a completely different front model in section 3.2?
4. Why report rAuA only for the robustness examples and not the other examples? The way this metric is reported, via a label on a plot, is also poor and would be much easier to read in a table.

### Soundness
3

### Presentation
2

### Contribution
2
