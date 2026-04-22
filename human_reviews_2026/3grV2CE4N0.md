# NeuroBasket: Interpreting Neuron Responses with Semantic Baskets

- Avg Score: 3.00
- Decision: Reject
- Scores: 2, 2, 4, 4

## Abstract
Deep neural networks excel across domains, yet their internal representations remain opaque. Prior approaches based on single neurons or non-hierarchical groups are limited by the distributed nature of concept encoding. We introduce Neurobasket, a framework that constructs semantically coherent multi-neuron groups through hierarchical clustering and natural language grounding. Neurobaskets enable set-theoretic reasoning, with unions revealing shared abstractions and differences highlighting discriminative cues. Experiments across convolutional and transformer models, trained on diverse datasets, show that neurobaskets yield stable and semantically aligned sets, while capturing prediction-relevant pathways. Qualitative visualizations further showed that grouped neurons correspond to coherent and localized concepts. Overall, Neurobasket provides a structured and compositional view of neural representations, extending beyond unit-centric or non-hierarchical explanations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Neurobasket, a new explainable AI (XAI) method for concept extraction in deep neural networks. In Neurobasket, activations from a specific layer are collected using images from a probing set and grouped through hierarchical clustering. To make sure the resulting clusters are meaningful, they are filtered based on several properties, including caption-embedding purity (L. 163).

These clusters are then matched with human-understandable concepts using vision-language and large language models. Finally, the clusters are discretized, meaning each one is linked to a specific set of neurons rather than a continuous direction in feature space. This is done by identifying which neurons are active for images that belong to a given Neurobasket.

The experiments show that Neurobasket identifies coherent neuron groups. They also demonstrate that unions and differences between Neurobaskets correspond well with human intuition.

### Strengths
- The paper is well written and easy to follow.
- The experiments cover multiple network backbones and datasets.
- The paper addresses an important problem and aligns with the current direction of the field, emphasizing the need to move beyond single-neuron explanations.
- The hierarchical approach is interesting, and it is appealing that unions and differences of concepts behave as intuitively expected.
- The appendix is detailed, providing essential experimental settings, and the authors also release the code, which supports reproducibility.

### Weaknesses
- **W1 Related Work.** In my opinion, the paper overlooks several important related works. Line 105 only mentions *FALCON* and *NeurFlow* in the context of MNEs, but omits studies that analyze feature space *directions* such as *CRAFT* (Fel et al., 2023) and *Sparse Autoencoders* (Cunningham et al., 2023). I also do not see a strong justification for why directions should fundamentally differ from sets of neurons.  
  Moreover, the idea of hierarchical representations has already been explored, for example in *Matryoshka SAEs* (Bussmann et al., 2025). The paper should either make a clear argument for why using neuron sets instead of directions is important, which is currently missing, or more explicitly position its contributions within the context of existing related work.

- **W2 Insufficient quantitative evaluation.** For many of the experiments, the paper only presents a handful of qualitative examples, which is not sufficient to properly support claims such as that Neurobasket "supports set-theoretic reasoning" (L. 86), that it offers "a more faithful and systematic view of neural representations" (L. 485), or that "explanations [...] are structurally organized and relationally grounded, offering deeper insights into how complex concepts are encoded" (L. 92). Also, following W1, relevant baselines for concept extraction through directions should be included for comparison.
Furthermore, the provided quantitative evaluations do not effectively assess the interpretability or utility of the proposed method. The analysis in Sec. 4 primarily checks whether the activations of the selected neurons are similar, which is desirable but not directly related to interpretability. In addition, the pruning evaluation in Sec. 5.3 is not compared to established methods for concept extraction, making it difficult to place the results in context.

- **W3 Hyperparameters are not thoroughly evaluated.** The method involves several hyperparameters, such as alpha and beta, but it is not clear how these affect the interpretability or overall results of the method. While the appendix (B.3) provides a brief evaluation of alpha, this only covers the proposed activation consistency metric and does not assess interpretability directly.  

- **W4 Filtering could remove important concepts.** The proposed method includes a filtering step that removes "uninterpretable" clusters (L.162). While this helps produce cleaner results, it raises the question of how faithful or complete the resulting explanations truly are.  


**Minor Weaknesses**  
- There are a few typos, e.g., L. 452, 425, and 459.  
- I only see experiments supporting the *union* and *difference* operations (Sec. 5.1 and 5.2); however, in L. 87, *intersection* is also mentioned as a contribution.  
- In Table 1, the standard deviation for *FALCON* is missing.
- The capitalization of the method’s name in the title is inconsistent with its usage in the main text (NeuroBasket vs. Neurobasket).

### Questions
- How does the method fit into the broader scope of concept extraction, including approaches based on direction analysis? If there is a clear unique selling point for neuron sets over directions, please provide supporting arguments or empirical evidence.  
- How can the interpretability and utility of the proposed method be evaluated, and how would it compare to existing approaches for concept extraction?  
- How sensitive is the method to different hyperparameter settings?  
- How many clusters are removed through the filtering step, and how does this affect the completeness of the resulting explanations?

While the paper is a nice read and presents interesting initial qualitative results, I feel it is still too far from the acceptance threshold, primarily due to the lack of quantitative metrics evaluating the interpretability and utility of the proposed method and comparing it to baselines using directions. Addressing these issues would probably require substantial additional analyses, which I consider beyond the scope of a standard review cycle.

I thank the authors for their effort and look forward to reading the rebuttal.

### Soundness
1

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes NeuroBasket, an interpretability framework that groups neurons into semantically coherent sets called "baskets" through hierarchical clustering and natural language grounding. It aims to move beyond single-neuron interpretability and non-hierarchical grouping methods.

### Strengths
- The paper articulates a real limitation in interpretability: single-neuron explanations fail when representations are highly distributed.
- Constructing a hierarchy over neuron groups and enabling set-theoretic operations (union/difference) is a useful conceptual advance.

### Weaknesses
1. The novelty of this paper is limited. Hierarchical grouping of neurons is not a new problem and has already been studied in previous works, such as (Wang et al., 2022), NeurFlow (Cao et al., 2025). The authors should clearly justify the key differences in their approach compared to other hierarchical grouping approaches. What are the new findings and justify the novelty.
2. For each basket, consistently active neurons are identified using β = 0.75. β (the hyperparameter for neuron set selection) is a key hyperparameter, but the choice of this value seems ad hoc and lacks theoretical or experimental justification. The authors should justify their choice of β.
3. Most experimental results are limited to comparisons with FALCON and a Random-index baseline. The authors should also include evaluations against other Multi-Neuron Explanation (MNE) methods, as well as representative Single Neuron Explanation (SNE) approaches (e.g., Clip-dissect, WWW). Such comparisons are essential to (1) justify the claimed improvements over traditional SNE methods and (2) demonstrate improvement compared to previous MNE frameworks.
4. The approach assumes that VLM captions and LLM summaries correctly reflect the underlying neuron or feature cluster semantics. However, VLM captions often contain hallucinations or bias from training data. So, the “human-understandable meaning” may not faithfully represent what the cluster actually encodes. Failure in this step undermines the entire interpretability pipeline.

- Wang, Andong, Wei-Ning Lee, and Xiaojuan Qi. "Hint: Hierarchical neuron concept explainer." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2022.
- Cao, Tue Minh, et al. "NeurFlow: Interpreting Neural Networks through Neuron Groups and Functional Interactions." The Thirteenth International Conference on Learning Representations. 2025

### Questions
Same as weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors proposed Neurobasket, a hierarchical framework for interpreting neural networks by grouping neurons into semantically coherent sets called *baskets*. 
It first applies hierarchical clustering using FINCH to activation features and grounds each cluster in natural language using captions from a vision–language model and LLM summarization. 
Neurons that consistently activate across a cluster’s images are then assigned to form stable basket representations. 
These neuron groups enable set-theoretic-like analyses (union, difference) to explore compositional structure in model representations. Experiments on ResNet and ViT models show that the proposed produces somewhat coherent, semantically aligned neuron sets that generalize across architectures and datasets.

### Strengths
- **S1. Practical and Coherent Integration of Existing Multimodal Tools for "Basket"**

The definition of a basket reflects a fair and coherent integration of existing techniques rather than introducing new algorithms. 
The framework combines hierarchical clustering with vision–language captioning and LLM summarization in a straightforward and practical manner. 
This design effectively leverages multimodal and language models to associate neural activations with interpretable semantic descriptions without adding unnecessary complexity. 
While not conceptually novel, the integration is technically sound and intuitively organized, making the framework easy to implement and extend to various architectures or datasets.

- **S2. Fair and Feasible Experimental Design Demonstrating Applicability**

The experimental setup is reasonably designed to demonstrate the applicability of the proposed framework across different architectures and datasets. 
In the absence of well-established baselines for multi-neuron interpretability, the inclusion of random-index and random-basket comparisons provides a reasonable reference for evaluating activation consistency and causal effects. 
This pragmatic approach ensures that the reported improvements of the proposed method are interpreted in context rather than overstated. 
Additionally, testing on both convolutional and ViT models helps illustrate the generality of the method, reflecting a fair and balanced effort to assess its robustness under diverse conditions.

### Weaknesses
- **W1. Limited Quantitative and Stable Evaluation of "Baskets"**

The concept of a "basket" in the proposed method--linking clustered neural activations with textual descriptions--offers an intuitive bridge between network activity and human-understandable semantics. 
However, because the semantic component is expressed only through LLM-generated text, it cannot be quantitatively compared or verified, leaving the evaluation of semantic coherence largely qualitative.

To enhance rigor and reproducibility, the authors could include quantitative analyses using semantic embeddings derived from basket descriptions. 
Embedding the captions or summaries into a multimodal space (e.g., Sentence-BERT) would allow measuring inter-basket similarity, quantifying hierarchical abstraction, and testing whether neuron-set operations correspond to meaningful semantic changes. 
Additionally, as the current grounding pipeline depends on captioning and LLM summarization, the resulting descriptions may vary with prompting or model choice. Embedding-based representations or text–image alignment checks could make the semantics more stable and verifiable.

- **W2. Conceptual and Empirical Ambiguity in Set-Theoretic Reasoning**

The paper highlights set-theoretic reasoning--particularly the union and difference of neuron sets--as a central interpretive feature of the proposed method. However, the implementation~(Fig. 4) appears more illustrative than strictly set-theoretic. 
Especially, the "union" between child baskets does not functionally generate the parent basket through the explicit combination of neuron sets; rather, the parent representation is obtained from an independently formed higher-level cluster. 
Consequently, the demonstrated relationship reflects qualitative alignment within the hierarchy rather than an actual compositional process where $S_p \approx S_a ∪ S_b$.

To substantiate this claim, the authors could quantitatively verify whether the union of child neuron sets reproduces or approximates the parent basket’s activation profile or semantic embedding. 
Demonstrating this correspondence across activation and semantic spaces would clarify whether hierarchical abstraction truly arises from compositional structure rather than coincidental clustering. 
Such validation would make the concept of set-theoretic reasoning both theoretically precise and empirically grounded.

- **W3. Ambiguity and Limited Validation in Qualitative Visualization**

Fig. 6 aims to illustrate how the proposed approach captures progressively abstract concepts across model depth, yet the notion of level remains only partly clarified. 
Fig. 2 indicates that level refers to the hierarchical depth within the clustering process, but in later figures the term is used alongside network layers (e.g., Layer 2-Level 3), blurring whether it denotes clustering granularity, architectural depth, or both. 
This conflation makes it difficult to interpret how hierarchical organization interacts with model structure to yield higher-level semantics.

Although the authors state that each image group corresponds to a specific basket, the alignment between the visual exemplars and their textual descriptions is not empirically validated. It remains uncertain whether the images faithfully capture the semantic content expressed in the accompanying summaries. 
To substantiate the claim that deeper layers yield more abstract and coherent representations, the authors could complement Fig. 6 with quantitative or human-based evaluations of image–text correspondence (e.g., CLIP similarity or participant agreement). 
Such analyses would strengthen the evidence that the proposed method captures genuinely human-interpretable hierarchical abstractions rather than relying on visual illustration alone.

### Questions
Most of my main concerns or questions have been outlined in the Weaknesses section.
I listed some additional questions here.

- Q1. If the textual grounding was primarily intended for interpretability, how might the authors ensure reproducibility in future implementations given LLM non-determinism?

- Q2. How sensitive are the results to hyperparameters such as the activation threshold ($\alpha, \beta$) used in neuron selection?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper addresses the problem of neuron grouping in Multi-Neuron Explanation (MNE) using hierarchical clustering. The paper enables set-theoric reasoning and hierarchical organization, which is original, using the neuron basket.

### Strengths
The paper provide concrete criteria on neuron activation and selection for each basket / group which previous works don't. The authors also propose a "consistency" metric to quantitatively compare the grouping quality between the basket and previous works. Furthermore, the proposal of set-theoric reasoning is original and interesting.

### Weaknesses
This paper lacks comparison experiment about the quality of clustering beyond "consistency":

1, It would improve my score considereably if the author can provide further comparison between ablating / pruning neuron groups from [1] and [2] in the experiment of "causal pruning evaluation". Showing that this paper is more effective at selecting neurons that is causally important for the prediction of the model. 

2, Beyond consistency, we may interest in other criterion such as "separatability", etc., between neuron groups. The consistency only shows  intra-level relation of one group, while separatability shows inter-level relation between multiple groups.

3, Could the author provide additional experiments to support the proposal of filtering metrics used in the main method (as discussed in Appendix B.1) on the quality of the neuron groups. Specifically, how do they affect the "consistency" and beyond, and how it is advantageous compared to previous works [1], [2]...

4, It is interesting to see the application of set-theoric reasoning in analyzing the effect of neuron groups on the model prediction, showing the usefulness of the proposal. For example, ablating higher level group (based on group hierarchy) affect more generalized concepts, while ablating set-difference would affect only a part of the concepts.

5, Minor comment: In the experiment measuring the "consitency", does the difference in the number of clusters for each method affect the outcome, for example, [1] uses 344 groups while the proposal method uses 10000 groups?

### Questions
Please refer to the Weaknesses section.

Citation:

[1]: Neha Kalibhat, Shweta Bhardwaj, C Bayan Bruss, Hamed Firooz, Maziar Sanjabi, and Soheil Feizi. Identifying interpretable subspaces in image representations. In International Conference on Machine Learning, pp. 15623–15638. PMLR, 2023.

[2] Tue M Cao, Nhat X Hoang, Hieu H Pham, Phi Le Nguyen, and My T Thai. Neurflow: Interpreting neural networks through neuron groups and functional interactions. ICLR, 2025.

### Soundness
3

### Presentation
3

### Contribution
2
