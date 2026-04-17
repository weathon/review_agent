# Tversky Neural Networks: Psychologically Plausible Deep Learning with Differentiable Tversky Similarity

- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6

## Abstract
Work in psychology has highlighted that the geometric model of similarity standard in deep learning is not psychologically plausible because its metric properties such as symmetry do not align with human perception of similarity. In contrast, Tversky (1977) proposed an axiomatic theory of similarity with psychological plausibility based on a representation of objects as sets of features, and their similarity as a function of their common and distinctive features. This model of similarity has not been incorporated as a general-purpose building block in deep learning, in part because of the challenge of incorporating discrete set operations.     In this paper, we develop a differentiable parameterization of Tversky's similarity that is learnable through gradient descent, and derive basic neural network building blocks such as the Tversky projection layer, which unlike the linear projection layer can model non-linear functions such as XOR.     Through experiments with image recognition and language modeling neural networks, we show that the Tversky projection layer is a beneficial replacement for the linear projection layer.     For instance, on the NABirds image classification task, a ResNet-50 with a Tversky projection layer trained from scratch achieves a 2.36 percentage point accuracy improvement over the linear layer baseline.     With Tversky projection layers, GPT-2's perplexity on PTB decreases by 7.8%, and its parameter count by 34.8%.     Finally, we propose a unified interpretation of both projection layer types as computing similarities of inputs to learned prototypes, along with a novel visualization technique. Crucially, Tversky's set-based representation enables the algebraic specification of semantic fields, which we illustrate with lexical and visual stimuli. Our work offers a new paradigm for neural networks that are not only more accurate and efficient, but also interpretable under an established theory of psychological similarity.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose how to use Tversky similarity (from cognitive psychology) instead of the geometric model of similarity, which is usually used in deep learning. They perform experiments on text and image domains.

### Strengths
* I really like the idea of using Tversky similarity to create layers for neural networks. Tversky’s theory is very logical and simple and I think that it has a huge potential for the neural network domain. 
* The paper is well-written and easy to follow. It is beneficial that the authors presented a small XOR example at the beginning. 
* The salience experiment was interesting.

### Weaknesses
* The paper lacks a clear related work section. While some works are described in the introduction, it would be good to include a dedicated section. Also, there are works that use Tversky’s similarity model or methods inspired by it for different purposes in the deep learning domain, and it would be beneficial to mention these works and acknowledge that using Tversky’s model of similarity is not completely new in deep learning - even if for different purposes.
* “This lack of interpretability, even in this simple domain without background textures, represents a significant limitation of prior approaches.” - the authors say that their approach improves interpretability. While it is visible that for MNIST (simple objects, with large differences) the prototypes look reasonable, I am not so sure that they would be that pronounced for the second dataset used in the study - regarding different breeds of birds. 
* For the vision experiments, it would be nice to include the results for a ViT as well - to represent two leading architecture families in vision.

### Questions
* “Some initializations of prototypes and features lead to convergence failure” - are normal and orthogonal initialization the ones that the authors refer to?
* Why didn’t the authors present the example graphical results for NABirds (like the ones in Fig. 2 for MNIST)?
* Also, see the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes an alternative to the commonly geometric formulation of similarity through using Tversky similarity. The authors propose a differentiable formulation of representations for this purpose, along with an implementation thereof (Tversky Projection neural layer). In addition, an approach for interpreting Tversky networks through visualization of projection layers in the data domain are proposed. Experiments on the XOR, Penn Treebank text data, and MNIST are performed to demonstrate the method.

### Strengths
- Clear and well-founded scientific motivation for the proposed novel methods. The idea of using Tversky similarity for deep learning is to my knowledge a novel contribution.
- Empirical confirmation of the proposed neuralized Tversky approach, in both synthetic experiments and large-scale studies
- The paper includes an honest and balanced discussion section and transparently raises limitations throughout the paper.
- The experiments appear reproducible and are for the most part clearly described (see below).

### Weaknesses
- While I like the motivation of Section 3.1 (XOR), the section and associated Figure 1 read rushed and can be confusing in its current state. I think the main take-home is that the Tversky projection layer can in principle learn the XOR function. I would suggest focusing on introducing the problem clearly (which is currently done in the caption), and then discuss some of the limitations encountered during optimization, i.e., convergence and hyperparameter selection.
- While I think the experiments are overall good, they remained superficial at times. Some quite interesting points were raised in the discussion, e.g. the idea of editing prototypes for increased robustness, which would have been great to demonstrate also empirically. 
- Some claims regarding interpretability are not clearly supported by evidence, e.g. “handwriting, such as lines and curves, more clearly than those learned by linear projection layers” (l. 413) and “Tversky Projection layers’s parameters are far more interpretable than the ones of the contemporary fully connected layer“ (l. 217). A more fair comparison could be to compare attributions in input space, e.g. [Mon18] as the Tversky Projection layers explicitly tie the projection to being decodable from the input. 
- The description of the proposed visualization technique in Section 2.5 should be more clear, e.g. through the use of formulas or a conceptual visualization, e.g. moving Fig. 5 to the main alongside a more clear description of how the method works. For example, the sentence “limitation that parameters specified in data-space are typically larger in size than their original counterparts, which increases the effective number of trainable parameters.“ (l. 211-213) is difficult to parse, i.e. how are the projection parameters specified, is the visualization technique trained or is this a post-hoc interpretability approach?


**Comment**
- I believe the proposed framework could be quite useful in the context of textual similarity, where it was shown that standard encoders and cosine similarity do not match human similarity judgments very well [Rei19]. Follow-up work on interpreting these dot-product embeddings could reveal quite simplistic feature matching strategies, even after aligning predictions to human similarity judgements [Vas24]. Here a Tversky layers may provide a more effective similarity readout.


**References**
- [Rei19] Reimers, N., & Gurevych, I. (2019, November). Sentence-BERT: Sentence Embeddings using Siamese BERT-Networks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP) (pp. 3982-3992).
- [Vas24] Vasileiou, A., & Eberle, O. (2024, June). Explaining Text Similarity in Transformer Models. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers) (pp. 7852-7866).
- [Mon18] Montavon, G., Samek, W., & Müller, K. R. (2018). Methods for interpreting and understanding deep neural networks. Digital signal processing, 73, 1-15.

### Questions
- Assuming contextualized models, would they be able to learn Tversky-type similarity?
- The notion of feature bank Ω could be communicated more clearly in Section 2.3. From Section 2.2 these are feature vectors $f_k$ and Ω parametrizes the function $f$, so $f$ essentially an embedding layer (as also mentioned more clearly later in Sec. 2.4.1).
- “For our vision experiments, no tuning was performed” (l. 459): No tuning *of hyperparameters* if I read Section 3.3 correctly, is that correct?

**Editing*
- “$α, β, andθ$” (l. 181)
- “fullly” (l. 217)
- “weights are initialized from ImageNet” (l. 327) - ImageNet -> ResNet?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new perspective on classification via prototype learning, where classifier decisions are made by taking the argmax over similarities to class prototypes.  Similarity is defined as an asymmetric match function between the object and a prototype over a learned feature bank whose dimensionality can be set.  While asymmetric similarity has been used in some prior applications (e.g., nearest neighbor classification based on engineered features), to my knowledge it has not been used in end-2-end CNN training, and the application is novel.

### Strengths
Strengths:
* Strong originality of entire prototype learning framework based on learned asymmetric comparison, which can be integrated in end-to-end training of various deep architectures. 
* The asymmetry weights, prototype dimensionality, and the feature bank are treated as learnable (free) parameters.
* Demonstrates improved performance on several benchmarks, including NABirds, indicating applicability to problems with hundreds of classes. While there is no demonstration of extension to thousands of classes, in this case I do not see as a limitation, because one of the central future applications of this training approach may be application to classification in relatively small domains with highly similar categories.
* A strength of the approach, whichcould be better discussed, is the potential to constrain the directionality of the asymmetry weights (e.g., setting β>α when omissions matter more, α>β when extras hurt more) depending on domain. For domains with low intra-class variance where 'core' features count (quasi necessary), penalizing omissions  of object vs. prototype might be more sensible; the converse for broader, more fuzzy categories.

### Weaknesses
(Minor) Weaknesses:

* It would be nice to have a small ablation experiment isolating the specific contribution of asymmetry by setting the asymmetry weights to zero and learning only the overlap (intersection) term in Equation 1.
* Can be extended to allow learning only the asymmetry terms with the overlap term to zero. Taken together we can understand if performance gains rely mainly on similarities or differences.

### Questions
It would be good to know if the Tversky head is effective as a classifier when attached to a backbone not trained for classification (e.g., a self-supervised backbone) and if in that case it would also outperform a baseline linear readout.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper challenges the geometric model of similarity (dot-product/ cosine similarity) largely used in deep learning, proposing instead Tversky's psychologically-grounded similarity measure that permits asymmetry and other violations of metric axioms observed in human similarity judgment. The key contribution is a differentiable version of Tversky's similarity and corresponding neural layers that can replace dense layers in standard architectures. The approach is based on the introduced dual representation of an object: objects are represented both as vectors and as sets of features from a learnable finite universe, where an object's set comprises those feature vectors with which it has positive dot product. This set-theoretic representation enables measuring similarity through feature set intersections and differences, forming the basis for Tversky Similarity and Projection layers. The authors demonstrate that: (1) a single Tversky layer can model XOR, impossible for linear layers; (2) Tversky layers can improve performance in GPT-2 language modeling (PTB dataset) and ResNet-50 image classification (MNIST, NABirds). Qualitative analyses reveal interpretable properties: salience scores (sum of dot-products of an object's vector with all features in its set) align with "goodness of form," learned (input-level) prototypes are human-interpretable, and semantic fields (e.g., adjectives, verb forms) emerge from set algebra. Additionally, they introduce a data-domain visualization method for projection parameters, though this approach scales poorly. 

The work is conceptually novel, well-written, and really promising but could benefit from modern architecture evaluation, clearer differentiability explanation, and stronger scalability analysis.

### Strengths
- Original and well-motivated idea connecting human similarity judgments to machine-learned representations. Clear novelty of First differentiable implementation of Tversky similarity for neural networks.
- The paper is well-written, clearly structured, and engaging to read. I learned a lot. 
- The XOR demonstration (Section 3.1, Figure 1) with its thorough validation convincingly shows that a single Tversky projection can model non-linear decision boundaries without composition with activation functions. This is a genuine advantage over linear layers.
- Demonstrates potential in both vision and language modeling contexts.
- Qualitative analysis really interesting.
- Acknowledges computational cost of data-domain visualization (Figure 5) and it is honest about the limitations.

### Weaknesses
- Gradient flow through indicator functions: Equations 2-5 rely on indicator functions [a·fₖ > 0] which have zero derivative almost everywhere. It's unclear how gradients propagate during training—are you using straight-through estimators, sigmoid approximations, or another approach? Providing implementation details or code would help to understand you work.
- Convergence of training: Tables 4-7 show convergence failure rates of 47-77% across many hyperparameter settings, with some producing NaN results. Understanding what causes these failures would strengthen the work. How did convergence look for the other main experiments?
- Evaluation on larger, more recent benchmarks would strengthen claims. Current experiments use MNIST (toy dataset), PTB (1M tokens from 1993), and NABirds (48K images with domain overlap with ImageNet birds), which are ideal for a proof of concept. However, testing on standard modern benchmarks like ImageNet, CIFAR-100, or WikiText-103 would better demonstrate competitiveness against well-tuned baselines.
- Similarly, validation on current architectures would show broader applicability. While ResNet-50 and GPT-2 small are solid choices for controlled experiments, evaluating on Vision Transformers (ViT, CLIP, DINOv2) or larger language models would support the "state-of-the-art" framing (L82).
- Multiple seeds are reported for XOR experiments, but it's unclear if main experiments used multiple runs. Would it be possible to comment on this issue or if available report the means and standard deviations?
- Tied vs untied feature banks remain unexplored. While prototype tying is ablated in Table 1, feature bank sharing is always enabled. Testing both configurations would be informative.
- Data-domain visualization might be impractical at scale. For NABirds with 555 classes, input-space parameterization would require 83.5M parameters (555×224×224×3) compared to 1.1M for standard approaches—a 75× increase. Could the authors comment on the scalability constraints and potential overfitting risks?
- Reporting accuracies: Using relative improvements (e.g., 24.7%) rather than absolute accuracy points can sometimes overemphasize small differences. Both would be informative.
- Methods like ProtoPNet and ProtoTree (prototype learning domain) seem related to the data-domain visualization contribution but aren't discussed in the introduction or chapter 2.5. Could the authors comment on that?

### Questions
- Please see questions already stated in weaknesses :)
- Methods:
    - As the author’s mention in the discussion, the size of  $\Omega$  is hyperparameter and needs to be tuned as everything else. However, do the authors have an intuition if there could be a relationship between the number of prototypes and the universe size?
    - Question about the set representations: A feature $f_k$ of the learnable finite universe is in the set of object $x$ if their dot-product is larger than zero. What does it then mean to include into this set representation a feature $f_j$  with $x\cdot f_j = 1e^-10$ and a feature $f_q$ with $x\cdot f_q = 1e^-10$?
- Experiments
    - Tversky layer can model XOR: In Tables 4-7 we can an observe rather small convergence ratios, could the authors comment on that?
    - Language-modeling: In Table 1, one can observe that only replacing the prediction head with a tversky head results in more features and parameters compared to replacing all projection layers with Tversky projections. This makes sense as the projection head itself must learn to represent all the semantic structure needed for predicting the next token. When all projection layers are replaced, the encoder and attention blocks co-adapt with the new similarity function, so the features can specialize efficiently, and fewer prototypes are sufficient. However, could you elaborate on the feature number and parameter number if we compare tversky-all with and without prototyp tying? Why do we observe such a difference?
    - Image classification:
        - Would it be possible to show some image classification results with some a pretrained Vision Transformer. E.g., a vision-encoder of a CLIP model, or Dinov3 model (available on huggingface)?
        - Why are the pretrained and frozen setting so bad in performance?

### Soundness
3

### Presentation
3

### Contribution
3
