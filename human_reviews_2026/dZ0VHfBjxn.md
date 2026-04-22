# Local Mechanisms of Compositional Generalization in Conditional Diffusion

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Conditional diffusion models appear capable of compositional generalization, i.e., generating convincing samples for out-of-distribution combinations of conditioners, but the mechanisms underlying this ability remain unclear. To make this concrete, we study length generalization, the ability to generate images with more objects than seen during training. In a controlled CLEVR setting (Johnson et al., 2017), we find that length generalization is achievable in some cases but not others, suggesting that models only sometimes learn the underlying compositional structure. We then investigate locality as a structural mechanism for compositional generalization. Prior works proposed score locality as a mechanism for creativity in unconditional diffusion models (Kamb & Ganguli, 2024; Niedoba et al., 2024), but did not address flexible conditioning or compositional generalization. In this paper, we prove an exact equivalence between a specific compositional structure (*conditional projective composition*) (Bradley et al., 2025) and scores with sparse dependencies on both pixels and conditioners (*local conditional scores*). This theory also extends to feature-space compositionality. We validate our theory empirically: CLEVR models that succeed at length generalization exhibit local conditional scores, while those that fail do not. Furthermore, we show that a causal intervention explicitly enforcing local conditional scores restores length generalization in a previously failing model. Finally, we investigate feature-space compositionality in color-conditioned CLEVR, and find preliminary evidence of compositional structure in SDXL.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the mechanisms underlying compositional generalisation in conditional diffusion models. To study this phenomenon quantitatively, the authors focus on the CLEVR dataset, consisting of synthetic images containing multiple objects of varying shape, size, and colour. The task is to train a conditional diffusion model to generate scenes with up to $3$ objects and to test whether the model generalises to scenes with more objects.

Building on previous work linking locality in learned representations to compositional generalisation, the authors hypothesise that generalisation arises when the target conditional distribution factorises over local components, leading to a decomposition of the score function into a sum of local terms. They empirically test this hypothesis by analysing the locality structure of learned scores in trained diffusion models.

The study further presents a controlled setting in which models fail to generalise to configurations with a large number of objects, showing that this failure correlates with a lack of locality in the learned score. Finally, the authors propose an architectural modification enforcing locality constraints, which restores compositional generalisation abilities.

### Strengths
- The topic is timely and highly relevant, addressing the important open question of compositional generalisation in modern generative models;
- The paper is clearly written, well-organised, and it communicates both the motivation and the technical content in an accessible yet rigorous manner;
- A key strength of the work lies in its combination of a theoretical perspective, linking compositional generalisation to the factorisation properties of the target conditional distribution (Conditional Projective Composition), with a carefully designed empirical analysis that tests these ideas on controlled datasets;

Overall, the paper exemplifies the kind of research the community needs more of: studies that propose interpretable theoretical frameworks for phenomena observed in deep learning, and that validate these frameworks through controlled, quantitative experiments.

### Weaknesses
1. The evaluation of length generalisation is rather limited. The paper primarily presents qualitative visualisations of generated samples, without defining a quantitative metric that can be systematically measured over an extensive test set. This makes it difficult to rigorously assess the generalisation performance.

2. The setup of the second experiment appears conceptually inconsistent. If the model is only trained on data containing a single conditional location, it is unclear how it could ever learn to generate objects at multiple locations. In this sense, the observed lack of generalisation may be an artefact of the training setup rather than a genuine failure of compositional reasoning, and therefore may not require a “fix.”

3. The locality of the score is a direct consequence of assuming a factorised data distribution. While this implies length generalisation, it is a property of the data, and it remains unclear under what conditions the model would naturally approximate this property and how the locality of the score could emerge during training.

### Questions
Weakness number 1 could be easily addressed by introducing a metric for the generalisation performance. Studying the evolution of such a metric in combination with locality during training could provide a path towards addressing weakness number 3. The latter would also require an example case where length generalisation does not occur beyond the setup called 'experiment 2' in the paper (see weakness 2).

### Soundness
3

### Presentation
4

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors study conditional compositionality in diffusion models and find it to be related to the 'local conditional nature' of the score function -- i.e. the sparsity of the dependency structure of the score on pixels and conditioners. They demonstrate evidence for this in a controlled toy setting, and further provide theoretical results supporting this finding. They also provide causal evidence (on the same toy setting) by introducing local conditionality and observing compositionality in a model where it previously did not exist.

### Strengths
- The paper approaches an interesting and important topic of compositionality in diffusion models. 
- They make a strong connection between prior work on locality and projective compositions to suggest a modified theory based on conditioners. 
- On a toy task their theory appears to match with experimental results.

### Weaknesses
- The figures are very complex, dense (and small) making them very hard to interpret for someone not intimately familiar with the prior work. 
- Until looking in the appendix, it was not clear what the rows and columns of Figure 1 are, making it hard to interpret, especially so early in the paper.
- There is heavy reliance on prior work (Kamb & Ganguli 2024, Bradley 2025) without sufficient description of these results to make the text self-contained. 
- Related work is deferred entirely to the appendix. 
- The discussion of 'feature space disentanglement' in the appendix is severely impoverished relative to the importance this related work holds. For example, the Spatial Broadcast Decoder (Watters et al 2019) appears to be very related in both in practice and in principle to many of the findings here. This problem is only made painfully more apparent by the authors choice to use the term 'disentanglement' to describe their own work.
- Figure 2 (Page 3) is not mentioned until Page 7 of the text. Rearrangement would improve readability.  Further, the organization of the paper in general is hard to follow given that the experiments depicted in Figure 1 are not discussed until page 7.
- The choice of how to construct length generalizing and non-length generalizing models in 'Experiments 1 & 2' appears somewhat arbitrary, and it seems like there could be many other confounding factors that contribute to the results. 
- The majority of the experiments are provided on the single toy setting with the single large scale experiments relatively unconvincing.
- It is not clear if the 'pixel-locality' of Figure 5 has any relation to the compositionally, or if it is just a correlation. 
- The authors appear to initially frame the paper around 'length generalization', but then make no mention of it for the larger scale experiments.

### Questions
- Have the authors explored other methods for construction 'non-length-generalizing' models, similar to the first experiments, to show that their theory still yields reasonable predictions?

### Soundness
3

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
2

### Summary
The paper studied compositional generalization in conditional diffusion models, focusing on length generalization in CLEVR. The authors proposed local conditional scores (LCS) and proved their equivalence to conditional projective composition (CPC). The authors then validate it with empirical results and show that enforcing locality causally restores generalization. Preliminary SDXL analyses suggest a similar structure in large-scale models.

### Strengths
1. The paper is well-structured.
2. The paper provides some theoretical results on compositionality and locality and verifies these results empirically.
3. The conclusions not only apply to toy datasets, but also somehow generalize to real-world datasets.

### Weaknesses
1. The evaluation is mostly qualitative, and some quantitative analysis would strengthen the paper.
2. Theoretical assumptions (disjoint pixel subsets) can be too strict for realistic data, as the authors mentioned.

### Questions
What is the difference between feature-space compositionality and disentanglement? It is known that generative models from the earlier StyleGAN to the later SD have disentangled latent spaces. If compositionality is considered equivalent to disentanglement, showing the block-structured correlation doesn't really add much merit, as this is well anticipated. Is there a way to enforce locality in SDXL to improve the compositional generation of realistic data, similar to what has been done with CLEVR?

### Soundness
3

### Presentation
2

### Contribution
2
