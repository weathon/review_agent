# Atlas-Alignment: Making Interpretability Transferable Across Language Models

- Avg Score: 4.80
- Decision: Reject
- Scores: 2, 8, 4, 6, 4

## Abstract
Interpretability is crucial for building safe, reliable, and controllable language models, yet existing interpretability pipelines remain costly and difficult to scale. Interpreting a new model typically requires costly training of model-specific sparse autoencoders, manual or semi-automated labeling of SAE components, and their subsequent validation. We introduce Atlas-Alignment, a framework for transferring interpretability across language models by aligning unknown latent spaces to a Concept Atlas — a labeled, human-interpretable latent space — using only shared inputs and lightweight representational alignment techniques. Once aligned, this enables two key capabilities in previously opaque models: (1) semantic feature search and retrieval, and (2) steering generation along human-interpretable atlas concepts. Through quantitative and qualitative evaluations, we show that simple representational alignment methods enable robust semantic retrieval and steerable generation without the need for labeled concept data. Atlas-Alignment thus amortizes the cost of explainable AI and mechanistic interpretability: by investing in one high-quality Concept Atlas, we can make many new models transparent and controllable at minimal marginal cost.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a framework, Atlas-Alignment, for transferring interpretable concept dictionaries, such as those learned by Sparse Autoencoders, across language models. They measure the utility of this framework for semantic search and model steering, finding that representational alignment across models allows for interpretability transferral.

### Strengths
- The application of the platonic representation hypothesis to interpretability is an interesting and active field of research right now.
- The authors test a variety of methods for cross-model alignment, including covariance, correlation, linear regression, Orthogonal Procrustes with row-wise L2-normalization, and SemanticLens. 
- The authors present promising results for using Atlas-Alignment for cross model steering.

### Weaknesses
- The motivation for this work hinges almost entirely on the idea that existing concept discovery methods (e.g. SAEs) are very costly to use; however, it is my understanding that this is not necessarily true. Prior work has noted that SAEs can be trained in a couple of hours on a single GPU, and automated labelling can be done with open-source models. Is there any benefit to Atlas-Alignment outside of this cost? Furthermore, Atlas-Alignment still requires the use of an existing Concept Atlas, which is generally just a pretrained SAE. Are there any reasonable alternatives to this? 
- The authors train their mappings with 500k samples, which seems equivalent to the amount of data required to train an SAE. Is it possible to scale this down to low data regimes?
- I also believe that the authors have overstated the difference between their work and existing literature that has argued for cross-model dictionaries or sparse autoencoders, such as [1, 2, 3] below. They simply note that previous "approaches require expensive training and can additionally suffer from high reconstruction loss" without providing empirical evidence for their training being less expensive, having higher reconstruction, or resulting in more effective steering. Providing quantitative comparisons with existing baselines across these axes would greatly strengthen this work. 
- The authors do not compare their framework to any strong baselines for any of the experiments. If they are claiming that Atlas-Aligned SAEs allow for cross-model steering, then they should at least compare against steering with SAEs trained for the target model to evaluate for any loss in performance. 


[1] Shared Global and Local Geometry of Language Model Embeddings. Andrew Lee, Melanie Weber, Fernanda Viegas, Martin Wattenberg.
[2] Universal Sparse Autoencoders: Interpretable Cross-Model Concept Alignment. Harrish Thasarathan, Julian Forsyth, Thomas Fel, Matthew Kowal, Konstantinos Derpanis
[3] Sparse Crosscoders for Cross-Layer Features and Model Diffing. Jack Lindsey*, Adly Templeton*, Jonathan Marcus*, Thomas Conerly*, Joshua Batson, Christopher Olah.

### Questions
- Why do the authors choose to max-pool over the sequences? 
- The authors write the following: "Steering with single directions often produces more incoherent generations, like endless repetitions of similar tokens. We instead combine several semantically related features to obtain more robust concept directions." How are the several related features found, and how are they combined? If steering one single direction is ineffective, does that not mean that that concept itself is poor? 
- How are multiple layers steered simultaneously? Why have the authors chosen to do this?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Atlas-Alignment, a method that constructs a human-understandable set of concepts (called Concept Atlas) from a foundation model and then aligns the representations from other models to the Concept Atlas for interpretation or steering. The Concept Atlas is created using sparse autoencoders. Then, a translation matrix between any model's representation and the Concept Atlas can be learned using any representational alignment method like Orthogonal Procrustes Translation. With this, they show that neurons representing similar concepts can be found across models and the models can be steered along those directions as well.

### Strengths
* The paper is fairly well-written and easy to follow.

* The proposed method is fairly simple, intuitive, seems novel and is effective.

* Qualitative and quantitative results seem to be fairly extensive (across models and datasets) and showcase the effectiveness of the method.

### Weaknesses
* The high-level technical novelty may be somewhat limited. For example, [W1] also proposes alignment of the representation space between two models, albeit for zero-shot classification. Or [W2] proposes linear alignment in a similar fashion but to produce individual neuron explanations for vision models. I do agree that Atlas Alignment's contributions are useful and noteworthy, but it would be good to discuss similarities and differences with these works given the high-level similarities. 

* In the proposed approach, the Atlas representations are translated into the subject model representations. Why not the other way? Apply the translation to the subject representations. That would make more intuitive sense because the Atlas representations are the human-understandable ones and it would be more useful to convert the subject model representations into Atlas representations.

### References

[W1] Moayeri et al., "Text-To-Concept (and Back) via Cross-Model Alignment", ICML 2023

[W2] Oikarinen and Weng, "Linear Explanations for Individual Neurons", ICML 2024

### Questions
Please address the questions/comments in the weaknesses section.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Atlas-Alignment, a framework that enables the transfer of interpretability between language models by aligning their latent spaces with a labeled Concept Atlas derived from sparse autoencoders (SAEs). Using lightweight representational alignment methods—particularly Orthogonal Procrustes—the framework allows an uninterpreted subject model to inherit semantic interpretability and controllable feature steering from an already interpreted model.

### Strengths
* The idea of transferable interpretability and connecting mechanistic interpretability with cross-model alignment theory is interesting. 
* The paper provides comprehensive and strong experimental results.

### Weaknesses
* This work relies heavily on the "Platonic Representation Hypothesis". Would this simple linear alignment still hold if the "subject model" and "foundation model" differed significantly in architecture, scale, or training data distribution?
* The paper invokes the Platonic and Linear Representation Hypotheses as justification but does not provide a rigorous theoretical analysis of their validity.
* Max-pooling removes sequence structure, potentially weakening the interpretability of attention-related phenomena.
* Ablation study is missing so it's unclear how translation quality scales with the number.

### Questions
Please see the weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a method to map SAEs trained on one model to another "subject" model. To this end, they assume that reprenentations of different models converge to a single representation space, as has been discussed in previous work. They create an SAE concept atlas (the human-labeled activations) and align the hidden representations of the subject models to that atlas using training-free representational alignment methods.  They show that with this alignment matrix, the subject model inherits semantic feature search from the first model (queries for “chess,” “dreams,” or “reddit comments”), and also concept steering (injecting translated concept directions into its activations), becomes possible.
To this end, the paper evaluates five concept atlas translation methods covariance, correlation, linear regression, SemanticLens-style, and orthogonal Procrustes) from Gemma-2-2B-based Gemma Scope 16k and Gemma Scope 65k SAEs across several subject models (Llama 3.1 8B Base, Instruct, R1-Distill). It evaluates translation quality (AUROC/AP), semantic retrieval (MRR/MPP, near-perfect), and model steering, where Procrustes performs best, e.g., 30–40% faithfulness improvements over baseline model steering).

### Strengths
- Reusing SAEs and various architectures is highly interesting. First to enable cost-efficient reuse and also for model steering
- The paper evaluates various alignment methods and several tasks/applications

### Weaknesses
- The paper relies on only a single atlas from Gemma scope. 
- It pools activations and as such loses some more specific information. 
- A human evaluation would strengthen the approach
- Much of the method (e.g., cross-model alignment, universal SAEs) relies on previous work, see related work

### Questions
Have you tried aligning to a model with different architecture, tokenizer, model sizes?
How stable is steering when we use token-based atlases or using atlases from different models?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a method for aligning language models representations to discovered concepts in a separate language models. 
By evaluating the linear transformation among two representation spaces, the resulting map transfers concepts of the SAE-based reference to the language models under study. 

The paper is mainly experimental and investigates the efficacy in transferring the concepts, the effect of steering, and semantic retrieval only by using the reference concepts of the SAE.

### Strengths
The idea behind the method is quite simple and straightforward: assuming the Platonic Representation Hypothesis, the authors propose to find a map $T_{s \to c}$ that maps model representations to activated concepts in the SAE reference. 
The method is quite simple and works good with orthogonal transformations. 

Overall, the results show the efficacy of the method for the task the authors have identified, including retrieval and steering. Moreover, the paper is well-written and easy to follow.

### Weaknesses
As far as I know, studying representational alignment through linear invertible transformations is a well explored field, where the most direct comparison is with [1].
Because of this, I cannot see much novelty in the proposal, if just considering using it with language models. While the results show efficacy in transferring the concepts of the SAE to another language models, it is not clear how this differs from previous proposal in [1]. 

Moreover, the paper rests on two fundamental hypotheses which have not yet been verified: the Platonic Representation Hypothesis, that even in its full acceptance does not necessarily imply linear transformations among representations, and the Linear Representation Hypothesis. For the latter, I think it is not necessary to assume it to have the method to work, since transferring concept of the SAE can be done in a linear manner even if the SAE concepts are not interpretable. 

Having accounted the gap in representational similarity between SAE reference and the subject models could be an interesting addition, which will make more concrete what limitations are inherited by the atlas. For similarity measures, I refer to [].

-----

[1] Latent Space Translation via Semantic Alignment, Maiorca et al. NeurIPS (2023) \
[2] Similarity of Neural Network Models: A Survey of Functional and Representational Measures, Klabunde et al., ACM (2023)

### Questions
I could not understand all details of the semantic retrieval setup that you tested. 
In particular, why are "subject model features i" used for the test? Do you mean choosing individual neurons i of the language models and then select inputs that maximally activate them? How costly is this selection and why not using sentences with ground-truth concepts annotations (e.g., the verb tense or the truthfulness)? What is the ground-truth in this setting?

### Soundness
3

### Presentation
3

### Contribution
1
