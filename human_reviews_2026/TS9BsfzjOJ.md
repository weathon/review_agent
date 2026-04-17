# Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction

- Decision: Reject
- Scores: 2, 6, 8, 4

## Abstract
Large Language Models (LLMs) have achieved impressive performance across diverse tasks but continue to struggle with learning transitive relations, a cornerstone for complex planning. To address this issue, we investigate the Multi-Token Prediction (MTP) paradigm and its impact to transitive relation learning. We theoretically analyze the MTP paradigm using a Transformer architecture composed of a shared output head and a transfer layer. Our analysis reveals that the transfer layer gradually learns the multi-step adjacency information, which in turn enables the backbone model to capture unobserved transitive reachability relations beyond those directly present in the training data, albeit with some inevitable noise in adjacency estimation. Building on this foundation, we propose two strategies to enhance the transfer layer and overall learning quality: Next-Token Injection (NTI) and a Transformer-based transfer layer. Our experiments on both synthetic graphs and the Blocksworld planning benchmark validate our theoretical findings and demonstrate that the improvements significantly enhance the model’s path planning capability. These findings deepen our understanding of how Transformers with MTP learn in complex planning tasks, and provide practical strategies to overcome the transitivity bottleneck, paving the way toward structurally aware and general-purpose planning models.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes to use multi-token predictions to improve language models' "planning" capabilities (e.g. path finding), which is an example of learning transitive relations.

Specifically, in addition to standard next-token prediction, they use a _transfer layer_ to predict a few future tokens; the same transfer layer is shared across different future positions.
- Most of the results are on a graph path finding setup following ALPINE (Wang et al. 24), where the model is expected to learn the true reachability matrix (i.e. whether a pair of nodes are connected) given paths observed in the training set. 

For theoretical results, they show with a simplified model that how predicting two-steps head (rather than only the next step) affects the model weights, in terms of the signs of gradients.

For empirical results, they apply multi-step prediction to improve the accuracy on the above synthetic path finding task and Blocksworld.
As implementation details, they use:
- next-token injection, which feeds in the ground truth tokens as input during training.
- a Transformer layer as the transfer layer, where the self-attention is across different dimensions.

### Strengths
- This paper applies multi-token prediction to improve the transitive reasoning abilities in language models, which is well-justified.
- The paper discusses both the pros and cons of multi-token prediction: while multi-token prediction helps capture higher-order reachability (Theorem 1,2), the paper also mentions that it may incorrectly bias the 1-step transition probability.
  - The paper claims the benefit of improving transitive reachability outweighs the risk of biasing 1-step transitions.
- The empirical results are promising:
  - For path finding, 2-token or 3-token prediction improves the accuracy especially on paths that require composing information from more than 2 training data sequences. 
  - For Blockworld, the paper shows that multi-token prediction provides some improvement over the 1-token baseline at different path lengths.
- The paper provides mechanistic study on how multi-token prediction affects the learned model.

### Weaknesses
I'm concerned about insufficient contribution.

- Theorem 1 and 2 (the results are about the signs of per-step gradients on a much simplified model) do not offer much beyond the intuitive explanation that the weight matrices are made to capture ground truth transitions.
  - This also comes with the cost of introducing notations (that are unnecessary for getting the key messages of the section) that impedes reading, so I'd suggest to keep the informal theorem in the main paper and move the details to the appendix.

- The experiment results are too limited to be convincing.
  - Multi-token prediction aligns well with the structure of the path finding task, so improved performance is expected. For Blocksworld, the amount of improvement is minimal, and even worse than the baseline when using a linear transfer layer. The paper does not demonstrate how much multi-token prediction would benefit more general setup.
  - For the path finding setup, the paper only considers Erdos graphs (with p=0.1) and not more structured graphs which are closer to applications (e.g. stochastic block models).

### Questions
- Sec 5.2, weight analysis: My understanding is that the transfer layer $W^T$ is not a Transformer layer in this case, since otherwise projecting $W^T$ doesn't make sense. Please clarify this in the writing.
- Table 3: It's not clear how to compare the value of "0.82" and "4.01". What are the ranges of the values? What's the variance in these values when averaged over several runs? Are normalization layer used? If yes, are these values before or after normalization layers?
- Table 1, 5: how many layers are used for the 1-token baseline? If it's 1-layer 1-head, then it's unclear whether the gain from multi-token prediction is due to the use of more Transformer layers (as transfer layers), or from multi-token training.

Minor: on line 112: has $N$ been defined?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper studies how multi-token prediction (MTP) can improve learning of transitive relations for path planning with Transformers. It analyzes a simplified one-layer Transformer with a shared output head and a linear “transfer layer” $W^T$ used to predict tokens multiple steps ahead. The analysis argues that: (i) WT is updated by the multi-step loss to approximate adjacency; (ii) gradients backpropagated through $W^T$  encourage the backbone to encode higher-order reachability; and (iii) some spurious adjacency may also be introduced. Two training/architectural tweaks are proposed: Next-Token Injection (NTI) that injects the ground-truth next-token embedding into the transfer input during training, and a Transformer-based transfer layer. Experiments on synthetic DAG path planning and Blocksworld show improvements over next-token training and re-implementations of “Meta-style” and “DeepSeek-style” MTP.

### Strengths
1. The research question studied is critical and timely, with a clean framing (transitivity bottleneck in planning).

2. The paper proposes a practical training tweak (NTI) that reliably improves results

3. The evaluations are comprehensive, including multiple MTP depths, backbone variants, and scaling across graph sizes.

### Weaknesses
1. Theorems are proven under a highly stylized setting: no positional embeddings, identity embeddings and output projection, fixed and manually set attention that always attends to the target (second position), linear FFN, single head/layer, and linear $W^T$ acting on logits.

2. The result that “$W^T$ learns adjacency” depends critically on $(W^M_{(i,d)}+W^V_{(j,d)})>0$. This is not guaranteed; if the sum is negative (or zero), the update direction flips (or vanishes). The paper partially acknowledges this by conditioning statements on positivity but then informally equates “positive-correlated intermediate nodes” with feasible intermediates.

3. The statement that $W^T_{(n−1)}$ “approximates the (n−1)-th power of the adjacency matrix” is not established. There is no proof of convergence or identification, and in the general case $W^T$ can absorb both adjacency and reachability information due to the non-uniqueness of the factorization with $W_t$ and $W_o$.

4. Transformer-based transfer layer description is unclear. The paper says it “leverages self-attention to model dependencies across dimensions” of $h_n$. A standard Transformer attends over the sequence dimension; with a single vector $h_n$ as input, self-attention degenerates. If you reshape features into a length-d sequence, please specify the exact reshaping, positional encoding, number of heads, masking, and why this is preferable to a simple MLP. As written, this component is conceptually confusing and under-specified.

5. Experimental details are missing. Decoding strategy, optimizer, learning rate schedule, batch size, epochs/steps, temperature, label smoothing, and regularization are not stated in the main text. Although a code link is provided, there is only a README file in it.

6. Only DAGs are used. Real planning graphs often contain cycles. An evaluation on cyclic graphs (with appropriate sequence formatting to avoid trivial loops) would strengthen claims.

### Questions
Suggestions:

1. Some figures use very small fonts. Adjusting them could make it easier to read.
2. Validate the gradient-sign predictions empirically by logging the sign of parameter updates vs the (p̂−pdata) terms during training.

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
5

### Summary
The paper shows that making LLMs predict multiple future tokens instead of just the next one helps them plan better on graph-like tasks.
They prove that this setup makes the model’s extra prediction layer basically learn multi-hop adjacency so it actually understands longer paths.

### Strengths
- Novel loss function to better train the model. Under MTP the transfer layer aligns with multi-hop adjacency, thus directly tying weights to structure. 

- Practical tweaks (Next-Token Injection and a Transformer-based transfer layer) are well-motivated and easy to slot into existing stacks. 

- Measurable gains on harder generalization (degree-2/3 paths) and Blocksworld, not just easy cases. 

- Learned transfer matrices progressively approximate true adjacency, thus giving a readable handle on structure.

### Weaknesses
- Scope is narrow: focus on DAG path-planning; little evidence for labeled/heterogeneous graphs or richer semantics (node/edge features).

- Scaling is lacking: experiments appear mid-scale; limited analysis for larger graphs, long paths, or real-world distributions.

- Lack of baselines: The experiments only compare against standard next-token training. There’s no evaluation versus other planning-aware or structure-inducing methods. So it’s unclear whether MTP is uniquely effective or just another form of multi-step supervision.

### Questions
- Permutation robustness: How stable are the learned transfer matrices under node relabeling or shuffled IDs? Any invariance tricks or results?

- Can the approach handle weighted/noisy edges or node/edge features (e.g., typed relations)? What changes in the theory?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates how multi-token prediction enhances the planning capability of language models. On a synthetic path-finding task, the authors provide theoretical analysis and empirical validation showing that MTP training enables models to capture transitive reachability relations. The findings are also validated on the Blocksworld planning benchmark.

### Strengths
The paper theoretically and empirically shows that multi-token prediction enables the backbone model to learn transitivity reachability relations, which standard single-token prediction fails to capture. This finding contributes to our understanding of how MTP may improve the planning capability of language models, which is very interesting to me.

### Weaknesses
1. The theoretical analysis focuses on one-step gradients without convergence guarantees. .
2. Effectiveness of MTP:
   1. In the simplified model, training with MTP allows the parameter $W^V$ to learn transitive reachability, whereas $W^M$ may capture spurious adjacency.   If I understand correctly, the degradation of $W^M$ could become more pronounced as the number of predicted tokens $n$ increases. It would be helpful for the authors to clarify the trade-off between these two effects as the number of predicted future tokens $n$ increases. 
   2. I am also curious whether a similar trade-off arises when applying MTP to more complex, real-world tasks.
   3. If my understanding is correct, MTP with a constant number of predicted future tokens $n$ can only enhance the model’s planning ability within that constant number of steps. This raises the question of whether MTP fundamentally improves planning capability. Previous work [2] also proposes training the model to directly predict future tokens ahead of the original next token. Compared to MTP, this method can predict a token arbitrarily many steps after the current token. I wonder how the two methods compare.
3. Novelty:
   1. The proposed Next-Token Injection and Transformer-based transfer layers appear similar to the multi-token prediction components described in DeepSeek-V3 [1]. It would be great if the authors could make the distinctions explicit if there are substantial differences. Otherwise, these components may not constitute a main novelty.
   2. Previous works already showed that multi-token prediction enhances planning capability. [3] 
4. The experiments are limited to synthetic tasks. It remains unclear whether MTP would significantly enhance planning capabilities in real-world or more complex reasoning benchmarks
5. It would be helpful to discuss related studies examining MTP’s role in planning, including [2, 3].
6. Minor comments: In Section 3.2, the notation $k$ is used both as a token index (l. 264) and as a node identifier (l. 284), which may cause confusion.

[1] DeepSeek-AI. DeepSeek-V3 Technical Report.

[2] Thankaraj et al. Looking beyond the next token.

[3] Bachmann and Nagarajan. The Pitfalls of Next-Token Prediction.

### Questions
1. Could the authors confirm whether my understanding is correct: By incorporating multi-token prediction, the matrix $W^V$ learns rechability relations beyond the observed $R^{\text{obs}}$. For instance, a 2-token prediction, if edge $k,k'$ appears in the training data, and path $k' \to t$ also appears, then $k\to t$ will also be enhanced in $W^V$, even when the full path $k\to t$ may not exist in the training data? However, for $W^M$, spurious adjacency may be introduced due to similar reasons. 
2. I wonder if the authors have observed degration in performance when increasing the number of predicted future tokens $n$ ? 
3. Could the authors clarify the main differences between the proposed NTI + Transformer transfer layer and the MTP architecture used in DeepSeek-V3 [1]?
4. l156–157: The paper states that “the two architectures are mathematically equivalent.” I am unsure why this equivalence holds when the transfer layers are nonlinear. If additional assumptions (e.g., linear transfer layers) are required, I suggest making them explicit.
5. l237: The main text argues that $W_{(j,k)}^V$ will increase when the model predicts a lower probability for $k'$ than the ground truth path $i\to k\to k'$. However, in Theorem 2, the condition is $\hat P_{i,j}(k') < P^{\text{data}}_{i,j}(k')$, where the node $k$ is not involved. My understanding is that the latter condition is the correct one. Can you confirm this or if I have misunderstood anything? 

Please also see the weakness.
[1] DeepSeek-AI DeepSeek-V3 Technical Report.

### Soundness
2

### Presentation
2

### Contribution
3
