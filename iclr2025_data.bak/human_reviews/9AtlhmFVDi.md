## Human Reviewer 1

### Summary
This paper explores the ability of Transformers to attend to spatial structures without relying on explicit structural modules.

By feeding linear embeddings of coordinates into Transformers, the Authors demonstrate that these models can approximate Gaussian spatial attention, enabling them to estimate and make use of spatial relationships. 

The Authors validate this approach in a sequence of steps, from simplified models to a protein language model similar to ESM1, concluding that a structurally enriched Transformer outperforms traditional graph neural networks (GNNs) in function prediction. 

This study contributes to using Transformers in spatial tasks, illustrating a new and useful capability of these models.

### Strengths
This paper seems to bring an original contribution, by demonstrating that Transformers can handle 3D structural reasoning on their own. 
This capability challenges the reliance on dedicated modules for spatial tasks that enforce symmetries and equivariance and introduces a new approach to protein structure modeling and other applications involving 3D data. 

The Authors provide a sound theoretical explanation of how Transformers can approximate Gaussian distance filters making use of coordinates embeddings, enabling them to encode distance relationships with the attention mechanism. In particular, it is appreciable their effort to make the findings accessible through the discussion of simple cases, providing a clear motivation for the approach.

The writing is great and makes the reading very easy and pleasant.

### Weaknesses
The quality of some of the plots is not excellent. For example, in Fig. 4 a-d plots are not very readable (tick labels are essentially invisible), and maybe a different strategy to convey the relevant information could be implemented.

### Questions
I did not get this comment "The version with coordinates also had a lower validation loss for the same training loss, and so the structural features learned early in training may be more robust to dissimilarity in sequence space."; it seems to me, from the plot that the two models never reach the same training loss. It may be trivial but can you clarify this passage please?

I suggest to update the references section because some of the papers cited have been published in the meantime (e.g, "Evolutionary-scale prediction of atomic level protein structure with a language model", by Lin et al.).

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 2

### Summary
This paper is concerned with analyzing the properties of Transformer-based protein language models, in terms of being able to capture structural properties such as physical distance once proteins are folded.  This idea is supported by theoretical and experimental developments that show that l2 distances (as in Gaussian models) seem to be the natural notion of distance that emerges in appropriate attention mechanisms.  Such Transformer-based protein language models perform well for the downstream task of protein molecular function.

### Strengths
Protein language models that operate on amino acid sequences are of significant interest for generative design, among other downstream tasks.  Likewise protein structure prediction, as in the recent Nobel Prize-winning work of AlphaFold*, is of central importance in biochemistry.  This work aims to demonstrate that protein language models themselves have some structural biology capabilities, which also helps with downstream tasks such as protein prediction; this provides significance.  The paper overall is also quite clear in what it does and doesn't do.

### Weaknesses
A swath of past work on the interpretability of protein language models, going back to Vig et al. in ICLR 2021 "BERTology meets biology" also show that attention mechanisms capture the folding structure of proteins, connecting amino acids that are far apart in the underlying sequence, but spatially close in the three-dimensional structure (among other results(.  See also references thereto.  It is not clear how much more extra novelty this paper provides beyond this existing line of literature, as no comparison/discussion is made. 

Downstream tasks are limited to just one.  Unclear whether the phenomenon of downstream utility is more general than that.

### Questions
What is new and exciting, as compared to "BERTology meets biology" and its ilk?

Are there other downstream structural biology tasks that benefit from the structural findings?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper (1) provides a theoretical explanation for how standard Transformers can learn to measure distance and perform structural reasoning, (2) shows that Transformers indeed learn Gaussian functions of distance and investigate efficient data augmentation methods
which can be used to learn SE(3), and (3) trains a protein masked token prediction model with coordinates and show that finetuning it for function prediction yields a model which outperforms structural GNNs.

### Strengths
1. The main idea of a paper is, indeed, interesting. It turns out that to some degree transformers without explicit SE(3) invariance can learn SE(3) invariant functions.
2. Experiment which confirm (1) are provided.

### Weaknesses
1. To which degree Equations 2,4 do hold? That is, what is the order of omitted terms?
2. Notation is ambiguous. Sometimes "x" is a vector, sometimes it isn't. For example, A1, A3.
3. I don't understand the purpose of a Section 3.2.3 "PROTEIN FUNCTION PREDICTION".
Using embeddings from pre-trained networks for protein property prediction is an established practice.
4. A very natural experiment is missing. You can take a pre-trained transformer which is presumable SE(3) invariant and shift/rotate coordinates, then check if its output changes.

### Questions
1. What is the problem with SE(3)-invariant GNN Transformers? You state that they tend to be memory-intensive, particularly
because attention is performed on edges, which grow as n^2 for fully-connected graphs. But transformers always have n^2 complexity, including yours.
2. Positional encoding in Eq. 5 is not standard. Does your study concern standard trigonometric positional encoding?
3. I don't understand A2. Can you provide a proof?
4. What does max(QKT) mean in Fig. 1 ?

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
In contrast with some existing structure prediction/reasoning methods, the authors argue that regular transformers are capable of sophisticated structural reasoning without the assistance of custom invariant graph neural networks. They demonstrate theoretically that hybrid structure-sequence transformers can learn to predict distance matrices. They then evaluate bare-bones transformers on that task as well as protein function prediction, showing parity with baselines.

### Strengths
I appreciate the attempt to simplify protein model architectures; these have always been relatively baroque, and so progress toward simpler but performant models would be welcome.

### Weaknesses
Overall, I'm unconvinced by the impact of this paper. There are also some question marks about the significance of some of the results. See below for detailed comments:

- > AlphaFold2 (Jumper et al., 2021) and ESMFold (Lin et al., 2022) preprocess protein sequences using Transformers to generate a representation which is used to condition an SE(3)-equivariant structure module to generate protein structures.

I'm going to quibble a bit with this characterization. While it's true that the structures are predicted by a custom structure module, AlphaFold2 also features a *distogram loss* independently of the structure module. While it's true that the Evoformer isn't a bog-standard transformer (it has GNN-flavored triangle attention modules, e.g.), these components were not critically important in the ablation studies for that paper (removing all "triangle" attention corresponded to a dip of < 5 GDT). I'd argue that one of the lessons of AlphaFold2, in comparison to AlphaFold1, was precisely that "(essentially) standard transformers are all you need" for structural reasoning. ESM2, a later development, is a standard transformer, also trained with a distogram loss, and does a pretty good job at contact/distogram prediction, as far as I know. What does this paper add that's not already present there? I think you need to do a much better job separating yourself from prior work. You shouldn't be trying to answer the question "can transformers perform structural reasoning?", since, as I've argued, that's already been established empirically elsewhere. At times you hint at a stronger version of that question: something like "are completely bog-standard transformers *all you need* for structural reasoning?" This is potentially interesting, but smells false to me: while not critically important, the fancy machinery in AlphaFolds 2 and 3 do seem to contribute to those models' edge over bog-standard ESM2. There's also nothing in this paper that would give us reason to believe that ESM2's architecture is not what is holding it back on the structure prediction front.
- How significant is the purported difference in loss values in Figure 2 (a)? Compared to Figure 2 (b), these values are all essentially zero.
- The confidence bars for the "Finetuned" results in Table 5 are absolutely massive---has there been some mistake? If not, there is no sense in which the MLP is "substantially better" than the alternative.
- Some parts of the paper are fairly unnecessary and should be in the appendix: Figure 3, for example, simply shows the benefits of the data augmentation procedure from AlphaFold3 without any modifications. I'm left wondering what is contributed to this study of whether transformers can learn to perform structural reasoning.
- In Table 5, the authors only compare to a baseline from 2021. The authors should add more baselines or do a better job explaining the reasoning behind their choice.
- "The input for these tasks could easily be augmented with more atoms without substantially increasing the memory footprint." - to be convincing, this statement would need to be accompanied by experiments showing that a) *predicted* structures work in this context, since there is no ground truth at the scale at which masked language modeling is typically performed or b) results showing that this sort of structure information can be fine-tuned into an existing PLM pretrained without structure information.

Minor comments (no bearing on score):
- The titles of both panes of Figure 2 are incorrect.
- You should explicitly list parameter counts for all models trained and evaluated in the paper.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4