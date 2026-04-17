# Chessformer: A Unified Architecture for Chess Modeling

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 4

## Abstract
Chess has played a uniquely important role as a testbed domain for artificial intelligence. Applying new architectures to improve absolute chess performance, and more recently to predict human moves at specified skill levels, has therefore garnered attention in the machine learning literature. Current approaches to these problems employ transformer models with widely varying architectural designs, and use unintuitive tokenization schemes that are not amenable to interpretability techniques, which hinders their applicability for teaching and human-AI interaction. We introduce Chessformer, a novel chess transformer model design that consists of an encoder-only model which processes chessboard squares as input tokens, instead of moves or the entire position, a dynamic positional encoding scheme that allows the model to flexibly adapt to the unique geometries present in chess, and an attention-based policy output design. We show that Chessformer advances the state of the art in all three major chess modeling goals: it significantly improves the chess-playing performance of a state-of-the-art chess engine, it surpasses the previous best human move-matching prediction performance with a much smaller model, and it enables substantial interpretability benefits. Our unified approach constitutes a broad advance across several important tasks in chess AI, and also demonstrates the benefits of carefully adapting transformers' tokenization systems, output systems, and positional encodings to reflect the structure of a domain of interest.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work introduces a transformer architecture meant to be trained over next-move prediction in chess. The two main changes are (1) GAB and (2) formulating the policy prediction over moves more simply (from start to end square). 

The authors take care to provide pseudonyms for models/works that are presumably related to their work. (Sometimes it makes it harder to understand the paper and the relative position of what they are trying contribute.)

### Strengths
* The writing itself is clear
* Figure 2 was interesting ; The tables 1,2 are clear
* Introduces a new intuitive way of framing how to format moves (an attention-based “source-destination” policy head)

### Weaknesses
* The first point raised in the abstract is only stated: "it significantly improves the chess-playing performance of a state-of-the-art chess engine." However, it does not describe this or give any information on this claim. While I am sympathetic to the challenge of anonymity issues, we cannot really take these 1-line statements into consideration. 
* The first (and primary?) set of results looks at the loss and accuracy of the models vs. another model (Figure 1) vs. ablations on human scores (Table 1). The differences while real and positive are not really shown to be material. GAB exceeds the Absolute approach by 0.16 percent. Even if this were statistically significant it is not obvious it is meaningful. Likewise, the Figure 1 results are similar. 
* Overall, it seems that the result is a strong one--a good architectural improvement--but that the demonstration of this based off of trust, not clear scientific ablations or clear documentation of what was done. This makes it hard to understand/evaluate the position of the paper.
* I did not find the interp. section compelling. There is a single figure comparison (cherry-picked? randomly-picked? representative?) in the first subsection. Likewise, in the next subsection they make similar one line note about SAE-like results. Again, it is not that I totally doubt the results are real. Rather, the results are insufficiently demonstrated.

### Questions
# Main Questions
* What is the primary goal of this work? It seems that the architecture is being sold as separate from the "state-of-the-art chess engine." When this is stated is this in reference to Table 3 (Table 3: Main results for raw playing strength.). 
* The second listed contribution, the human-emulation matching, again, seems better, but marginally so, to the point it is unclear (or not sufficiently). The third, again, is the interpretability, and it seems again likely/possibly interesting, but just not shown.
* Do you report or comment on the stat. range or std of the scores reported in your tables? Anything to help contextualize the results would be helpful. The accuracy deltas being so small make it hard to appreciate the results. Are we near a ceiling of performance? Is the task hard or is the data stochastic? 
* Q: "Empirically, we find that GAB is a key driver of these gains." Note that in Table 1, GAB also requires more FLOPS and param. Where is this clearly demonstrated?


# Minor Questions
* Q: In section 3.2, "we concatenate representations of the past 7..." How are the concat? I am reading this as "stacking" the information onto each of the 64 tokens. What are the final dimensions?
* Q: Are the embeddings for the weak/strong players learned/updated? (Any comments on the relatively large dimension for this 128dim; the rest of the embeddings is 12 for the pieces + some other auxiliary information?) It is unclear what the actual/final architecture and dimensions of the model are.

# Minor Notes
* Table 2. The parameter sizes are very different from those before. I think grouping together the different models and sizes actually used into a clear section would help.
* As shown in Table 5.3 --> As shown in Table 3,
* L429 "finder"
* L247 "achieves its largest gains at the highest calibers of strength *shows that* Chessformer" --> "suggests that"

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a novel transformer architecture specialized to playing chess (e.g. approximating optimal play, or for imitating human play). A key novel component is GAP ("geometric attention bias"), which essentially adds an additional component computed by an MLP to the pre-softmax attention scores, as a more dynamic alternative to traditional positional encodings. The new "Chessformer" architecture also uses a new type of policy head. Experiments for both optimal and human-imitating play show clear performance gains over other architectures, even at lower parameter counts and inference costs. The new architecture also has properties that make it easier to interpret, and the paper takes initial steps towards understanding the functions of different model components.

### Strengths
- The architecture contains clever ideas and is well-motivated for the domain of chess.
- The empirical results are very impressive, demonstrating clear gains over previous work and ablations, even when using over an order of magnitude less inference compute than prior work. The evaluation methodology is convincing.
- I feel that there are lessons to be learned beyond only chess. The paper is a great example showing that a simple off-the-shelf transformer baseline can be decisively beaten in non-language domains with a more specialized architecture. And the motivation for GAP could apply to other domains as well where positions/distances aren't well-described by a static approach.
- I found the SAE interpretability results (in appendix B.1) highly intriguing.

### Weaknesses
- While I think some of the lessons from this paper could generalize to other domains, the target audience may still be a bit narrow.
- The description of the architecture (in particular GAP and the policy head) could likely be made easier to follow with some figures showing those novel components

### Questions
1. Does GAP fully replace any positional encoding? If so that seems interesting/surprising, since if I understand correctly, GAP only directly affects attention scores, so MLPs and attention value vectors would not directly receive any positional information. Did you experiment with e.g. both GAP and an absolute positional encoding? And do you have guesses for why putting positional encodings directly into the residual stream of the transformer (rather than only attention patterns) isn't important for performance?
2. How cherry-picked are the two SAE features shown in the appendix? E.g. did you look at 100 features and these were the only ones this interpretable, or are half the features roughly this clean?

### Soundness
4

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
4

### Summary
This paper presents Chessformer, a transformer-based model for chess that unifies engine play, human move prediction, and interpretability on board attention. The model represents chess positions using 64 square tokens and introduces a Geometric Attention Bias (GAB), which is a dynamic positional encoding that adapts to board geometry. It also uses an attention-based policy head aligned with chess’s “from–to” move structure. Experiments show Chessformer outperforms prior models like Allie in both playing strength and human move prediction while being more efficient and interpretable.

### Strengths
- The introduction of GAB is a creative and well-motivated innovation that aligns the model’s attention with the spatial and semantic structure of the chessboard.
- The experiments are very comprehensive, covering both engine and human benchmarks. The ablation studies also show consistent performance gains.

### Weaknesses
- While the Geometric Attention Bias is central to the paper’s contribution, it is only presented in pseudocode within the appendix. A main-text figure illustrating its structure, input–output flow, and how it modulates attention across the board would make the concept far more intuitive and strengthen readers’ understanding.
- The description of how Lichess data were sampled and balanced across Elo levels is brief and lacks specific counts or sampling ratios, which may limit reproducibility.

### Questions
- How was the Lichess 2023 dataset processed in practice? How many samples per Elo range were used, and what criteria guided the downsampling to balance skill levels?
- You mentioned "Chessformer models mainly adapt the GAB biases to global positional features like the game stage (opening, middlegame, endgame), rather than the locations of individual pieces." Why is GAB able to recognize different game stages?

### Soundness
4

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper introduces Chessformer, a novel transformer architecture specifically designed for the domain of chess, which significantly improves move prediction and playing strength over prior approaches. Chessformer makes multiple domain-specific architectural improvements: encoding the 64 board squares as tokens, using an attention-based “source-destination” policy head instead of naively one-hot encoding all possible legal moves, and using a Geoemtric Attention Bias (GAB) to convey positional information. Evaluations on move prediction and game-playing show that the Chessformer matches or outperforms prior approaches at a fraction of the cost.

### Strengths
The proposed approach is sensible and appears to work well in practice. The empirical validation is comprehensive and shows that Chessformers match or outperform prior work at a fraction of the scale and cost. The paper conducts ablations to show that the Geometric Attention Bias outperforms traditional positional encoding schemes. The Geometric Attention Bias and the “source-destination” attention head are novel and well-suited to chess. The proposed architecture modifications promise to facilitate domain-specific interpretability research by being more suited to the geometry of chess. The paper is well-written and easy to follow.

### Weaknesses
The main weakness of this work is that it is restricted to chess and, therefore, likely to be of marginal importance beyond the chess-ML community. Chess has served as an important testbed for many ideas in AI; however, this paper’s contribution is to make chess-specific adaptations to Ruoss et al. (2024) to obtain better performance. While the paper does a fine job at that, it is not quite clear to me what anyone outside of the chess community can learn from this work.

Given the above, the primary contribution of this paper should be to advance the state-of-the-art in the narrow subdomain of searchless chess modeling in a _reproducible_ manner, i.e., by releasing the code, model parameters, and/or the dataset. To the best of my knowledge, the paper does not address any of these aspects, unlike prior work (Ruoss et al., 2024).

The paper makes the unsubstantiated claim that “Chessformer significantly improves the chess-playing performance of a state-of-the-art chess engine” and “contributed to match wins over Stockfish in multiple computer-chess tournaments”. However, there is no empirical evidence to back up this claim.

The paper claims to use a novel encoding scheme; however, it is quite similar to the one proposed by Ruoss et al. (2024), who utilize an expanded FEN notation, i.e., 64 board states (and some additional information, which this paper also encodes). The main difference between the two approaches is that Ruoss et al. (2024) only feed the current board state to the transformer, while this paper proposes to concatenate it with the previous 7 board states.

There are a few typos:
* L269 “name changed”
* L352 “absolute position”
* L355 “Table 3”
* L376 “A recent line”
* L429 “much finer interpretation”

### Questions
* How did the paper arrive at concatenating the current and the 7 past positions? It would be interesting to ablate this particular architectural choice.

### Soundness
4

### Presentation
3

### Contribution
2
