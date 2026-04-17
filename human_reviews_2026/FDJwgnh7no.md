# A Two-Character Change in Transformer Architecture Promotes Ideal Token Geometry

- Decision: Reject
- Scores: 6, 4, 4, 2

## Abstract
We hypothesize that in the optimal geometric configuration of token embeddings for transformer classifiers, tokens should collapse to single points according to their classes, and these points themselves should exhibit Neural Collapse. We study whether current transformers achieve this configuration through principal component projections, cosine similarity measurements, analysis of variance on token embeddings, and Neural Collapse measurements, and find that they fall far short of the conjectured ideal. To address this, we introduce a simple modification to attention that brings token embeddings markedly closer to the conjectured configuration and yields consistent performance improvements across benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes a Laplacian-based modification to the transformer attention mechanism that encourages token embeddings to evolve toward a beneficial geometric structure called Neural Token Collapse (NTC). By interpreting attention as diffusion on a graph, the method introduces a Laplacian “head” that smooths token representations along meaningful variance directions rather than averaging them, improving both representation structure and classification accuracy. Experiments on CIFAR-10, CIFAR-100, and ImageNet-1k show consistent performance gains over standard Vision Transformers. The authors connect their approach to related concepts such as neural collapse, rank collapse, and oversmoothing in graph neural networks, suggesting that controlled token collapse can enhance model efficiency and separability rather than hinder it.

### Strengths
- Originality: There are a few points of novelty in this paper: exploiting token collapse as an improvement of the transformer architecture, rather than an issue, is surely a clean idea that I have not seen before. Along this line, the physical interpretation as graph Laplacian diffusion makes it more grounded to known concepts.
- Clarity: The problem is stated clearly in the introduction and raised questions and goals are well referenced when addressed along the whole text.
- Quality: This paper proposes an hypothesis grounded on the understanding of how transformers works, implements a modification to the architecture that verify the hypothesis and, importantly, as a result improve the performance of the model. This is how a paper on interpretating neural networks should be structured.
- Significance: Principled improvements to transformers architectures are highly relevant in a moment where available human training datasets are being saturated.

### Weaknesses
1. The discussion in 2.1 and 2.2 has been heavily motivated by Figure 2. While the figure neatly conveys the idea of how a single attention block modifies the token’s relative positions, it is not clear to me that this would work as neatly in high-dimensional spaces. Some empirical support or citations for the statements made in 2.1 and 2.2 can improve the presentation.
2. The experiment part seems a little bit underdeveloped: the authors consider only one vision model and three datasets, which are used both for training and for showing results. It would also seem natural to discuss what would be the result of applying this method to language models/next token prediction tasks.
3. Presentation-wise, the authors might make a slight effort in improving how they present their experimental results (see questions below for more detail).

### Questions
1. While the authors give some details about training, since the method involves a different paradigm, some comparison on the training metrics with the base model (like just the number of epochs) could be useful?
2. The mixing of attention and laplacian seems a bit arbitrary. Appendix D seems to discuss more mixing trials, but I wonder if there is a way of framing this in a more principled way? Would this mixing have to be adapted at each task?
3. Some discussion on extending this framework to next token prediction/autoregressive tasks can be helpful for a more impactful paper. Do the authors think such an extension is possible, i.e., is NTC ideal for next token prediction? Some empirical validation by training small transformers using Laplacian heads would be great.
4. Some figures might benefit a little bit more cosmetics: 
- Figure 3 has two diagrams with identical labels and no reference to top/bottom in neither text nor caption
- Figure 4 some labels are cut.
- Figure 5 the legend for “M” and “W” is explained in the caption but might be made more explicit in the plot. Y labels are not in latex despite using equations.
5. In the classification performance part, the datasets used are the same ones used in training. Would it be possible to see this on an unseen dataset? 
6. This paper https://arxiv.org/pdf/2408.15417 seems relevant and I haven’t seen it in the relevant works sections.

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
2

### Summary
This work presents a method for obtaining better token embeddings, as motivated by the neural collapse phenomenon in the existing literature. Extensive empirical evidence is presented to backup the idea that current token embeddings are suboptimal, and a simple fix can be done to mitigate this. In particular, experiments on vision transformers are done on CIFAR classification tasks, as well as probing into the geometry of the existing embeddings.

### Strengths
- The work provides an interesting perspective on how to improve token embeddings, which can only be beneficial transformer/language model pipelines.
- Experiments showing the existing geometry of embeddings and neural token collapse are shown, as well as improvements on image classification tasks.
- The visualizations given are very useful for the reader.

### Weaknesses
- Experiments are run on image classification tasks only; it would be interesting to see if such improvements hold for generative models as well.

### Questions
- On the classification tasks, it seems that the improvement is not extremely significant. I wonder if it is strictly necessary to have better token embeddings (e.g. the ones that satisfy this analogue of neural collapse).
- Following up on my above comment (see weaknesses) for generative models: in current LLMs there are positional embeddings that encode the tokens in addition to the token embeddings themselves; is there any reason to believe that the marginal improvement towards the NTC regime will be significant enough for the case where we apply positional embeddings?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper proposes the concept of Neural Token Collapse (NTC) as a desirable geometric property for token embeddings in transformers. They show that standard transformer architectures fail to achieve the so called ideal token geometry. To address this, the authors introduce a a simple modification to the standard attention mechanism where the output PV is changed to V−PV. The paper demonstrates through experiments on CIFAR-10, CIFAR-100, and ImageNet that this modification not only brings the token geometry closer to the ideal NTC but also yields consistent and significant improvements in classification accuracy.

### Strengths
-The proposed idea seems to be novel and simple. It simply modified the attention output from PV to V - PV. It offers a practical and low-effort way to improve existing transformer models without adding parameters or significant computational cost.

-The proposed method achieves meaningful performance improvements across all tested datasets, including a 5% absolute improvement on CIFAR-100. 

-Interesting theoretical insights. The paper provides a compelling theoretical interpretation of the proposed change by connecting it to graph theory.

### Weaknesses
- It is unclear that why the proposed NTC is the ideal token geometry. 

- Limited Scope of evaluation. All experiments are conducted on image classification. While the theory is general, the paper lacks evidence of its applicability to other domains, most notably Natural Language Processing. It is an open question whether forcing this kind of NTC would be beneficial for other tasks.

- The paper proposes mixing standard Attn heads with the new Laplacian L heads to further improve performance. However, the strategies for this mixing (e.g., "1P", "3P", "Mix-Depth") feel somewhat heuristic and are not as well-motivated as the core Laplacian idea itself.

### Questions
The authors are encouraged to provide evaluation results in NLP.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes neural token collapse (NTC) as an ideal token geometry for transformers in the setting of classification. Inspired by the neural collapse line of work, this work focuses on the specific case of transformer architectures and proposes simple architectural fixes to the attention mechanism to promote NTC. Experiments are conducted on image classification tasks to verify the effectiveness of the proposed method.

### Strengths
1. The proposed method is very simple and can be incorporated into standard transformers easily.
2. The geometric framing is straightforward and the ANOVA approach makes the results easy to analyze and interpret.
3. The connection to diffusion over graphs is interesting and aligns well with the experimental results.

### Weaknesses
1. It is very questionable whether achieving zero variance among tokens within the same sequence is truly desirable. Note self-attention precisely promotes dynamic weighting between tokens as a mechanism to propagate information. Removing the variance among tokens leads to a trivial weighting and effectively makes self-attention no better than a mean aggregation. To test this, one can add another baseline with just a simple mean operator to perform token mixing, without any self-attention.
2. The method only makes sense for classification, and I find it hard to extend the method to the vast amount of tasks transformers do well in: causal language modeling, dense segmentation, masked image modeling and other self-supervised visual modeling.
3. In experiments, the authors only tested the model with the hybrid approach described in Sec.2.3. It is not clear what the performance would be if using only the proposed method in Sec 2.2. This makes it hard to understand the behavior of the proposed method.
4. Related to the previous point, the results are not truly convincing, as the classification performance gain on ImageNet-1k is very small. It is questionable whether the proposed fix alone is worth incorporating into any existing ViTs.

### Questions
Please see above.

### Soundness
2

### Presentation
3

### Contribution
1
