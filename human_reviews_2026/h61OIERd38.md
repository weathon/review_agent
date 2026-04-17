# Hierarchical Concept-based Interpretable Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Modern deep neural networks remain challenging to interpret due to the opacity of their latent representations, impeding model understanding, debugging, and debiasing. Concept Embedding Models (CEMs) address this by mapping inputs to human-interpretable concept representations from which tasks can be predicted. Yet, CEMs fail to represent inter-concept relationships and require concept annotations at different granularities during training, limiting their applicability.
In this paper, we introduce *Hierarchical Concept Embedding Models* (HiCEMs), a new family of CEMs that explicitly model concept relationships through hierarchical structures. To enable HiCEMs in real-world settings, we propose *Concept Splitting*, a method for automatically discovering finer-grained sub-concepts from a pretrained CEM’s embedding space without requiring additional annotations. This allows HiCEMs to generate fine-grained explanations from limited concept labels, reducing annotation burdens. 
Our evaluation across multiple datasets, including a user study and experiments on *PseudoKitchens*, a newly proposed concept-based dataset of 3D kitchen renders, demonstrates that (1) Concept Splitting discovers human-interpretable sub-concepts absent during training that can be used to train highly accurate HiCEMs, and (2) HiCEMs enable powerful test-time concept interventions at different granularities, leading to improved task accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces multiple novelties to make classification more interpretable. Basing on CEM embeddings, Concept Splitting discovers new, hierarchical subconcepts based on SAEs. Then, HiCEM integrates these sub-concepts into the CEM structure. To evaluate, PseudoKitchen is introduced, a granular dataset with concept control. Also, for imagenet, a user study is conducted to evaluate the meaningfulness of the discovered subconcepts.

### Strengths
* This work integrates SAE's concept discovery capabilities into the CEM framework. While there have been attempts at CBM + SAE combinations, the application of SAEs on the CEM embeddings is a neat idea, as these embeddings naturally lend themselves to such an analysis. How to integrate the discovered concepts into CEMs is nontrivial and it seems the authors engineered a way that works nicely.  
* In interpretability research, human user studies are the gold standard, but nearly never done. As such, I appreciate the authors' efforts to conduct a user study, whose results have also positively affected my perception of this work. In general, the results are extensive and span many datasets and different axes of evaluation, which support the claims made in the paper.
* Also, the introduction of PseudoKitchens might be a valuable addition to the literature, as it provides realistic, controllable, sample-specific concept control and could help the field get away from class-based concepts.
* I appreciate that reproducible code is provided.

### Weaknesses
* My main concern of this work is that it bases on CEMs. I do not consider CEMs truly inherently interpretable, mainly due to their embedding-based design, which has also been demonstrated by [1,2] that show that CEMs contain a lot of leakage. Therefore, any derivative work also has questionable interpretability in my view. For example, the embeddings of the discovered sub-concepts might contain a lot more information than just with regard to their sub-concept. In this line of thought, the added complexity of the sub-embeddings makes the model more opaque, thereby even harder to interpret. There is a trend of arguing that intervention effectiveness implies that there is no (strong) leakage, however, due to the RandInt strategy at training time, this argument does not hold in my opinion. I can be convinced otherwise, if intervention effectiveness was retained without RandInt training for both CEM and HiCEM.
* Understanding the architecture of HiCEM in Sec. 4.1. took a while. I wonder whether it might be easier to understand by replacing the flowtext with more structured formulas or by using a pseudoalgorithm. Also, I just realized there is no Sec. 4.2, and there shouldn't be any single subsections, but that's unimportant here.
* The paper reiterates that related work does not model concept relationships. I am referring to the sentences "Furthermore, they do not consider the relationships between concepts; instead, they treat all concepts as independent variables" and "However, the modelling of these relationships addresses a gap in previous concept-based architectures, advancing the state of the art in concept-based explainability". In my opinion these statements are misleading. There is a considerable line of literature that tackles precisely this problem [3,4,5,6] and given the focus of this work on modeling these concept relations, this related field should at least be mentioned.

[1] Parisini, Enrico, et al. "Leakage and interpretability in concept-based models." arXiv preprint arXiv:2504.14094 (2025).

[2] Almudévar, Antonio, José Miguel Hernández-Lobato, and Alfonso Ortega. "There Was Never a Bottleneck in Concept Bottleneck Models." arXiv preprint arXiv:2506.04877 (2025). 

[3] Havasi, Marton, Sonali Parbhoo, and Finale Doshi-Velez. "Addressing leakage in concept bottleneck models." Advances in Neural Information Processing Systems 35 (2022): 23386-23397.

[4] Vandenhirtz, Moritz, et al. "Stochastic concept bottleneck models." Advances in Neural Information Processing Systems 37 (2024): 51787-51810.

[5] Xu, Xinyue, et al. "Energy-Based Concept Bottleneck Models: Unifying Prediction, Concept Intervention, and Probabilistic Interpretations." The Twelfth International Conference on Learning Representations.

[6] Singhi, Nishad, et al. "Improving intervention efficacy via concept realignment in concept bottleneck models." European Conference on Computer Vision. Cham: Springer Nature Switzerland, 2024.

### Questions
I invite the authors to address any of the aforementioned weaknesses, albeit I deem it unlikely that it will positively affect my score.

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
The paper presents Hierarchical Concept Embedding Models (HiCEMs), extended from Concept Embedding Models (CEMs) to represent inter-concept relationships with various granularities. The work also introduces Concept Splitting, a technique for automatically discovering finer-grained subconcepts from pretrained concept embeddings without requiring additional annotations. Together, these methods enhance interpretability, reduce annotation costs, and enable hierarchy concept interventions while archives comparative performance. The paper also introduces a new synthetic dataset, PseudoKitchens, with precise ground-truth concept annotations and spatial localization.

### Strengths
1. The HiCEM archives comparable performance comparing to previous methods.
2. The extensive experiments show that the alternative concept splitting method (clustering) also achieves comparable performance to the original design (SAE).
3. The paper proposes an additional synthetic dataset, PseudoKitchens, a controllable scene with precise annotation.

### Weaknesses
1. What is the actual benefit of the hierarchical design? Although the experiments show that HiCEM can leverage subconcepts to intervene in the model and achieve better performance on the CUB dataset, in most cases, its performance is comparable to that of CEM.
2. Although the paper includes a user study to evaluate the connections between subconcepts and their parent concepts, it remains unclear whether these subconcepts also contribute to improving the overall interpretability.
3. In the current version, the authors focus only on a two-level hierarchical concept structure. It would be interesting to explore deeper hierarchies and examine the relationships across multiple levels.

### Questions
1. It would be helpful if the authors could provide qualitative comparison results between different methods, which would make the improvements in interpretability more intuitive and easier to observe.

### Soundness
2

### Presentation
2

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
The authors introduce HiCEMs, a variant of concept embedding models (CEMs) that 1) supports inter-concept (hierarchical) relationships and 2) in addition to the usual supervised concepts, discovers additional sub-concepts in an unsupervised manner using a sparse autoencoder (SAE).  HiCEMs are compared against a number of CBM-like competitors and the interpretability of sub-concepts the SAE discovers is evaluated through a (small scale) user study.

### Strengths
- **Novelty**: HiCEMs combine CEMs with concept discovery.  While the two
  ingredients are well known, their combination (and especially the
  rather intricate sub-concept module that is introduced to combine them) look
  novel to me.

- **Quality**:  Using SAEs/clustering for concept discovery in a
  well-structured embedding space (obtained via per-concept supervision) is
  reasonable, but see below. The choice of research questions is good and the
  experimental setup (choice of datasets and competitors) is sound.

- **Clarity**: the text is generally well written, except for the sub-concept
  modules, see below. The related work is relatively complete.

- **Significance**: the idea of pairing SAEs with concept annotations will
  probably inspire follow-up work. This work also partially fills the gap
  between concept-bottleneck models and neuro-symbolic architectures, which is
  interesting.  The new dataset is also a welcome addition.

### Weaknesses
While I am generally positive about the paper, I'd like to raise some issues I found.

- **Quality**

    - My main concern is that SAEs do not provide any sort of guarantee.
      Recent works (which may or may not be under submission at ICLR, I have
      not checked) suggest they *can* recover the underlying generative
      concepts provided these are sparse, but in general the jury is still out
      on this, to the best of my knowledge.  Using SAEs is fine, but the
      authors should be careful in not overselling their ability to recover
      good concepts.

  - The interpretability of SAE sub-concepts rests on the user study.  This is
    however quite small scale (20 participantes; by the way, this information
    is important and belongs in the main text; currently it is "hidden" in Appendix E).
    Are these results statistically significant?

  - I also have a (relatively modest) issue with the structure of the user study itself.  The thing is the sub-concepts are named using english words through a "discretization" step (specifically, first english words are mapped to clip embeddings and then hicem sub-concepts via a probe, then sub-concepts are named accordingly). It is possible the sub-concepts don't match the name exactly, but only very approximately, which means that this step essentially "cleans up" the discovered sub-concepts.  To what extent do the results rely on this step? How dissimilar are the names and the associated sub-concepts?

- **Clarity**

  - My main concern is the description of the sub-concept modules, which is
    quite dense and consisting of pure free-form text.  It should be unpacked,
    ideally with the help of equations, to make it clearer and more precise.

  - The CLIP-based automatic naming procedure should be described in the main
    text, as it is integral to the method -- without it, sub-concepts cannot be
    properly interpreted by human stakeholders, breaking self-explainability.

  - I am confused about the term "sub-concept supervision" used in p 7 onward.
    Does it refer to the fact that Concept Splitting is absent? If so, it'd
    make sense to either rename it "sub-concept weak supervision" or to call
    the baseline HiCem-NoSplitting or somesuch.

  - Table 2: "Even better results might be obtained by manually selecting
    high-quality sub-concepts, and naming them manually instead of using our
    crude automatic naming method with CLIP." Very likely, but this is
    equivalent to requiring expert sub-concept supervision - you claim to
    tackle settings in which this is not available.  Moreover, depending on how
    this is obtained (UI-wise), experts could misidentify concepts, so there is
    no a prior guarantee that this would work better than the CLIP probe
    procedure. I think this sentence should be removed.

 - The MNIST-ADD dataset was not introduced by LeCun et al.  Please, fix the
   reference.

 - It should be more clear in the text that the "SHAPES" dataset is similar
   to the relatively well-known dSprites dataset.

### Questions
Feel free to comment on any of my observations.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces two new tools for interpretability specifically targeting the hierarchical nature of concepts/conceptual reasoning. To this end, they introduce a "discovery" interpretability method, as well as a new concept-based deep model to leverage this hierarchical structure. In a nutshell, the discovery method proposed aims to find novel subconcepts from "high-level" concept embeddings using sparse autoencoders. The hierarchical concept embedding model (HiCEM) introduces a new sub-concept neural module that explicitly models a concept starting from its sub-concepts. The authors provide a series of programmatic experiments, as well as human user study to demonstrate their new tools.

### Strengths
- *Originality and significance*: while the idea of hierarchical concepts is not new per se, the authors create a novel solution specific for Concept Embedding models which seems compelling.
- *Quality*: overall, the experiments seem reasonable to support the authors claims. In particular, I like the inclusion of user studies and intervention experiments.
- *Clarity*: the paper as a whole is generally understandable.

### Weaknesses
- *Originality*: as mentioned above, the idea of hierarchical concepts is not new, e.g. "Hierarchical Concept Discovery Models: A Concept Pyramid Scheme" Panousis 2023. Could the authors expand how their work relates to previous work specifically in the context of hierarchical models of concepts?
- *Significance*: I am completely sold on the idea of concepts being hierarchical (or at the very least not being completely independent). That being said, this does not necessarily mean that modeling them is necessary for better interpretability or intervenability, for the simple reason that often full transparency may be too overwhelming for a person to fully process what a model is doing. Do the authors have in mind a specific real-world use case where their model and concept splitting idea brings a practical advantage compared to other SOTA models?
- *Clarity*: I understand the limitation in space, but I think it would be beneficial to expand on the evaluation of the discovered (sub-)concepts. More below.

### Questions
1) Regarding clarity of evaluation of sub-concept discovery: 
  - L. 310, do all the provided datasets have concept-subconcepts pairing? If not, how is "between the discovered sub-concept labels and their potential parent-concept-associated matches in the bank." (L.315) evaluated?
  -  Isn't it possible that maybe a (sub-)concept is correctly discovered but for some reason is not in the left-out pool of concepts from the concept bank?
  - How many concepts are left-out? How is the number of features/sub-concepts selected during discovery with SAE? if the number of left out concepts and "features" are not the same, is this taken into account in evaluating the discovery process? I.e. is this considered a multi-label classification problem? 
2) How does concept splitting depend on the quality of the top-level concepts? On first thought, I would perhaps think that if the concept embeddings from the pretrained CEM are not good enough, this would hinder the discovery of sub-concepts? 
3) From your description, it seems to me that concept splitting does not necessarily depend upon the concepts from CEM. If so, would it then make sense to create some sort of synthetic evaluation experiment, where you run the concept splitting on predefined "perfect" top-level concepts. For example, it shouldn't be a problem to define the concept embeddings as essentially one-hot encoded vectors, I think?
4) As a minor nitpicky suggestion, I would maybe try to expand on why concept splitting is such a pivotal contribution from the authors. Personally, for how it is phrased now, concept splitting sounds like an application of concept discovery methods to pre-trained concept embeddings instead of directly to a latent representation of a sample. So if we put it like this, concept splitting sounds more like a necessary step to then train good HiCEM rather than a key standalone contribution.
5) Based on the authors results, the main advantage of HiCEM vs normal CEMs is essentially the higher intervenability as demonstrated on, eg., the CUB dataset. Could the authors advance hypotheses regarding why this is the case? Is it because CUB and PseudoKitchens are more complex datasets and there actually exists complex parent-subconcept relationships that need direct modeling? Based on this, I believe the limitations section could be expanded by commenting on when it is expected that concept splitting+HiCEM is an extra effort that is worth to do.
6) It would be interesting to have some intervenability human user study to strengthen even more the paper. Something along the lines of "How would you change this explanation to get this prediction?"

### Soundness
3

### Presentation
3

### Contribution
2
