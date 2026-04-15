# Hierarchical Self-Attention: Generalizing Neural Attention Mechanics to Hierarchy

- Decision: Reject
- Scores: 5, 8, 8, 5

## Abstract
Transformers and their attention mechanism have been revolutionary in the field of Machine Learning. While originally proposed for the language data, they quickly found their way to the image, video, graph, etc. data modalities with various signal geometries. Despite this versatility, generalizing the attention mechanism to scenarios where data is presented at different scales from potentially different modalities is not straightforward. The attempts to incorporate hierarchy and multi-modality within transformers are largely based on ad hoc heuristics, which are not seamlessly generalizable to similar problems with potentially different structures. To address this problem, in this paper, we take a fundamentally different approach: we first propose a mathematical construct to represent multi-modal, multi-scale data. We then mathematically derive the neural attention mechanics for the proposed construct from the first principle of entropy minimization. We show that the derived formulation is optimal in the sense of being the closest to the standard Softmax attention while incorporating the inductive biases originating from the hierarchical/geometric information of the problem. We further propose an efficient algorithm based on dynamic programming to compute our derived attention mechanism. By incorporating it within transformers, we show that the proposed hierarchical attention mechanism not only can be employed to train transformer models in hierarchical/multi-modal settings from scratch, but it can also be used to inject hierarchical information into classical, pre-trained transformer models post training, resulting in more efficient models in zero-shot manner.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper derives self-attention as a gradient descent step to minimise the variational upper bound of entropy and formulates a new hierarchal self-attention by changing its energy function. There are experiments comparing this new hierarchal self-attention and the standard self-attention on multi-modal data and zero-shot transfer learning.

### Strengths
1. The notation is clear and problem well motivated. 

2. The authors make an effort to provide background information, which could help those unfamiliar with the topic to understand the context.

### Weaknesses
1. The derivation of self-attention does not consider the value matrix which is a crucial component. Despite this, another paper [1] has already derived attention from the update rule of hopfield networks which is relatively similar to this work. 

2. The formulation of hierarchal self-attention (HSA) is unclear and the key methodological steps are missing, making it difficult to fully understand the approach. 

3. The experiments are limited and do not provide strong evidence of the method’s effectiveness. They are performed on small datasets like IMDB and it is not convincing that the datasets chosen have a hierarchal structure that requires a new attention mechanism. The multi-modal experiment only has marginal improvements and there is a significant drop in accuracy on the zero-shot transfer learning. 

4. While there might be less flops used for HSA, there is doubt that the actual run time is lower due to the algorithm used in HSA. 

[1] Ramsauer, Hubert, Bernhard Schäfl, Johannes Lehner, Philipp Seidl, Michael Widrich, Thomas Adler, Lukas Gruber et al. "Hopfield networks is all you need." arXiv preprint arXiv:2008.02217 (2020).

### Questions
1. What is the novelty in the derivation of self-attention in this paper in comparison to hopfield networks? Why are the differences between the gradient update in the entropy minimisation and self-attention not important? 

2. Please clarify clearly and simply the exact steps required for HSA and provide the run time during training and inference. 

3. Can HSA be applied to larger and more complex standard datasets like imagenet or C4/Wikitext-103? And can it be shown that these datasets benefit from using a hierarchal structure?

### Soundness
2

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
The authors present a new way to formulate the attention mechanism of Transformer architectures based on a formal mathematical construct that is able to capture/model the hierarchical structure of multi-modal multi-scale data with different underlying signal geometries in a consistent and general manner.

### Strengths
**Originality & Significance:**   
- Motivation of the method based on a clear observation of shortcomings in previous heuristic approaches, leading to the introduction of a well-formalised mathematical foundation to model different signal geometries in one consistent framework
- Proposed method applicable to wide range/combination of modalities due to the very few assumptions that are made (nested signals very flexible in terms of exact composition / instantiation of internal nodes)  
- Authors also provide a way how their contributions can be incorporated in already trained ‘conventional’ Transformers, further expanding the applicability of their findings

**Quality:**   
- Contributions are well-placed within the wider area of research, good overview of related methods
- Good range of experiments presented that mostly support the underlying intuition and argument of the paper

**Clarity:**   
- Contributions and underlying motivation clearly stated in intro 
- Explanations well-supported through the two main Figures
- The paper is well written and easy to read and follow; longer proofs moved to the appendix, and the paper therefore provides a good level of depth to follow without interrupting the reading flow

### Weaknesses
_TLDR; I do very much like the underlying idea, but have some questions and concerns that I’d like the authors to clarify & address._

- Potential limitation to the structural assumption made during derivation of ‘conventional’ softmax attention and how this affects the transferability of the presented concept/insights onto current Transformer architectures is not (sufficiently) discussed – see questions
- Comparison of obtained results with a ‘current’ Transformer of the same size (which does not necessarily adhere to the constraints defined in ll. 260ff) – see questions
- Experiments exclusively on minuscule scale for Transformer models

### Questions
**Main concerns, questions & potential improvements:**

- Four main constraints/variations that deviate from current Transformer layouts are set out in lines 260ff to match the attention formulation;   
  Is this only required to show the mathematical equivalence with the ‘popular’ softmax self-attention, or actually used throughout the experiments? In either case, I’d like to know to which extent this affects the generality of the authors’ proposed approach, specifically:
  - No separate value projection, but use of keys as values: This might severely limit the expressivity of the architecture; Do the authors see this as a potential drawback, and could this be fixed/accommodated for? 
  - Post-norm instead of pre-norm for LayerNorm: Pre-norm significantly improved training stability, which is especially important for larger architectures. As the authors’ experiments mainly tackle small scales, is there a possibility this might become restrictive? And could this be accommodated somehow?  

- Results reported for non-hierarchical approach:
The authors compare their HAS to FSA, i.e. flat self-attention; Does this still include the constraints mentioned before? If so, how would a ‘current’ Transformer perform on these tasks (w/o these modifications)? 

- L 317 f: _'Approximation of attention-weight between any leaf node in A and any leaf node in B by one value'_:  Is there an underlying intuition why this would make sense? I can see this at higher levels, but when thinking about one modality being an image and another one a sentence describing it – I would expect sub-parts (e.g. tokens) of each modality to definitely have different attention weights, as some words will refer to specific parts of the image? (same applies to audio, etc.) 

- Replacing self-attention in pre-trained transformer-based models by HSA w/o retraining:
As the structure of the HSA is essentially different, how is this performed? Assuming a relatively recent architecture with Q,K,V and pre-norm – how would HSA fit in there? Would the attention matrix simply be replaced while still using the same projection matrices and layer structure?   
Also: Replacing degrades performance, as discussed towards the end of Sec 5; – would this be ‘solved’ when training an entire (larger) model from scratch with HSA? -> Also see next question: 

- Scale of used models: Table 1 shows experiments for model with 1.2M params, which is extremely small (even for small and efficient models). This goes up to 12M later on, which is still extremely small.    
-> Do authors have any impression how their approach scales to larger models? I do understand computational constraints, but some insight would be very helpful for the reader! (especially given that even highly-efficient models working on multiple modalities are typically orders of magnitude larger).  

---

Additional comments:    
- L 228 ff: Would the dimensionality of the key and query variables really need to be $\mathbb{R}^d$, i.e. the same dimension as the input? I assume key and query space could be of any dimension?     
-> Note: This distinction might be important when thinking about the actual dimensionality of key/queries within a Multi-Head Self-Attention, where each head is of lower dimension than the input/embeddings

---
---
### Update / Adjustment (rebuttal phase):  
Given the lack of response by the authors to my and other reviewers' questions, I'm slightly downgrading my current score from 6 to 5. I'd encourage the authors to respond as there is still time left for discussion.

---
---
## Update  post-rebuttal phase:  
Given the provided rebuttals, I have raised my score (as discussed below) to 8. While there are some remaining weaknesses around the empirical validity/usefulness compared to other hierarchical approaches, I do think this work passes the bar given the insights it provides.

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
2

### Summary
The paper proposes to adapt the transformer architecture to multi-modality signals or signals presented at different scales by a hierarchical self-attention (HSA) mechanism for nested signals. The paper first generalizes the classical Softmax attention mechanism (traditional transformer) by representing it as a solution to an entropy minimization problem of a simple signal (without hierarchy). Then, based on this insight, the paper derives a theoretically grounded mechanism for calculating self-attention within nested signals. The authors evaluate the proposed approach in language and multi-model news classification problems and show improved performance. In addition, the authors show that the proposed approach can be utilized for fine-tuning, where a specific form of hierarchical information is revealed at test time - the simple self-attention operation can be exchanged for HSA, leading to efficient calculations at test time without the need for re-training.

### Strengths
* The paper proposes a novel, theoretically grounded mechanism for hierarchical self-attention that elegantly generalizes the classical Softmax self-attention for flat signals.  

* The paper is well-written and easy to follow. The example of website representation demonstrates the use cases of the proposed algorithm and provides convincing motivation. 

* The paper includes a thorough discussion of the related work.

### Weaknesses
**Limitations:** The authors did not address all the limitations of the work. For example:

* Adding the inductive bias of the signals’ structure during training can constrain the model to output only signals with this specific structure. In addition, it’s not straightforward to indicate the structure of the signals in every dataset.

* The authors acknowledge that the algorithm is not readily parallelizable on standard GPUs and requires dedicated hardware for acceleration - strangely, this is omitted from the main paper and only mentioned in the appendix.


**Comparisons:** 

Although the authors mention that previous works attempt to incorporate hierarchy and multimodality within transformers (based on ad hoc heuristics), the proposed method is compared only to flat self-attention and DeepSet (work from 2017).

### Questions
1. Please address the aforementioned weaknesses.

2. In section 5.3 the authors mention that when a pre-trained flat self-attention is replaced with HSA and fine-tuned, the results match the performance of the original flat self-attention, with improved computational cost. Why doesn’t incorporating the indicative bias improve the performance as well? Given that the model can be fine-tuned until convergence.

3. The authors mention that the leading algorithms usually incorporate tricks such as using multiple loss functions to achieve SOTA performance. Is it possible to incorporate the same tricks with the HSA models? 



minor typo: Line 254:  Eq. equation 4 →  equation 4

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
In this paper, the authors aim to obtain a hierarchical self-attention mechanism. They aim to show that they have a new concept of a nested signal. Further, they aim to use the notion of entropy minimisation to derive a form of hierarchical attention that they claim is optimal. They provide evaluations on a few datasets and mainly provide comparisons between flat attention and the proposed hierarchical self-attention. Further, the comparisons with DeepSet are provided. The main focus is on theoretical validity

### Strengths
* They aim to consider the idea of hierarchical self-attention using their definition of a nested signal and a hierarchical self-attention mechanism that is derived theoretically using ideas of entropy minimization
* They provide comparisons with the flat attention mechanism and the hierarchical self-attention mechanism is observed to be superior to the flat attention
* Finally, they consider the zero-shot generalisation of a pretrained flat attention transformer to a hierarchical transformer and provide comparisons

### Weaknesses
* The hierarchical self-attention transformer is only bench-marked against flat self attention and Deep Set. The authors have observed that various hierarchical self-attention methods have been proposed in the literature. If as they claim, the proposed hierarchical self-attention is indeed better than other proposed hierarchical self-attention, then it is imperative to benchmark their work against such works.
* The notion of self-attention was principally to reduce an inductive bias - for instance the translation invariance inducted by convolutional neural networks. By re-introducing an explicit induction bias, is that not detrimental? Can the analysis be done based on this notion of inductive bias?
* There have been substantial number of works that have considered harmonic analysis based operators such as adaptive Fourier neural operators, scattering vision transformers and many such neural operator based attention mechanisms that have demonstrated superior performance with extensive bench-marking. In contrast, the proposed work does not benchmark or compare it with other such methods for inducing structure in a transformer.
* While the authors have described their approach to hierarchical self-attention, it is not clear how conceptually their approach differs from previous hierarchical self-attention. The formulations have been considered theoretically in this work, but, the resulting loss function is quite close to the loss function used previously. So, an analysis and comparison with previous hierarchical approaches would be expected

### Questions
The clarifications requested would be any clarifications that the authors can provide with respect to the weaknesses mentioned above.

--
Update post rebuttal - In view of the clarifications provided, I have updated my score from 3 to 5.

### Soundness
3

### Presentation
3

### Contribution
2
