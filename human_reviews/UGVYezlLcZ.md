# On the Optimal Memorization Capacity of Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Recent research in the field of machine learning has increasingly focused on the memorization capacity of Transformers, but how efficient they are is not yet well understood.
We demonstrate that Transformers can memorize labels with $\tilde{O}(\sqrt{N})$ parameters in a next-token prediction setting for $N$ input sequences of length $n$, which is proved to be optimal up to logarithmic factors.
This indicates that Transformers can efficiently perform memorization with little influence from the input length $n$ owing to the benefit of parameter sharing.
We also analyze the memorization capacity in the sequence-to-sequence setting, and find that $\tilde{O}(\sqrt{nN})$ parameters are not only sufficient, but also necessary at least for Transformers with hardmax.
These results suggest that while self-attention mechanisms can efficiently identify input sequences, the feed-forward network becomes a bottleneck when associating a label to each token.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper examines the memorization capacity of Transformers. The authors claim that Transformers can memorize labels with $O(\sqrt{nN})$ parameters in next-token prediction for sequences. They argue that parameter sharing reduces input length sensitivity. In sequence-to-sequence tasks, $ O(\sqrt{nN})$ parameters are shown to be necessary and sufficient for hardmax transformers. While self-attention identifies input sequences effectively, the feed-forward network seems to be a limiting factor.

### Strengths
1. The paper addresses the important problem of quantifying the memorization capacity of transformers.
2. The authors present theorems that cover a wide range of tasks for which transformers are used, making the analysis comprehensive.

### Weaknesses
1. The sketches of the proofs in the paper were somewhat difficult to follow without referring to the full proofs. Improving the presentation could help a wider audience better understand and appreciate the work.
2. While the paper is theoretical, and experiments may not be essential, including experiments to support the claims would strengthen the work.

### Questions
1. I don't completely follow the reasoning behind the statement: "the primary bottleneck is not the contextual mapping of tokens, but rather the feed-forward layers’ capacity to map this token-level contextual information to labels." Clarification on this point would be helpful.

2. If the feed-forward layer is identified as the bottleneck, is there an approach to mitigate this through architectural changes or modifications?

3. Are the data assumptions too restrictive? It feels surprising that the results hold if the attention layer primarily averages, especially when compared to findings in studies such as (a) and (b), despite the differences in setting.

4. Would it be possible to include experiments, even toy examples, to support some of the paper’s claims? Such experiments could significantly enhance the paper's impact.

I am open to increasing the score for the paper based on the authors' responses and feedback from other reviewers.

References:

a) Rajaraman, Nived, et al. "Transformers on Markov data: Constant depth suffices." arXiv preprint arXiv:2407.17686 (2024).  

b) Makkuva, Ashok Vardhan, et al. "Local to Global: Learning Dynamics and Effect of Initialization for Transformers." arXiv preprint arXiv:2406.03072 (2024).

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studied the memorization capacity of transformers in two tasks, namely next-token prediction and sequence-to-sequence prediction. The paper showed that, by construction, a transformer with $\tilde{O}(\sqrt{N}log n)$ layers can memorize $N$ data points of length $n$. The paper further showed that a transformer with $\tilde{O}(\sqrt{nN})$ layers can memorize $N$ data points in the sequence-to-sequence prediction problem. A lower bound is provided to demonstrate the near-optimality of the upper bound.

### Strengths
1. The results seem promising and strong, because both upper and lower bounds are provided.
2. The studied problem and method provide interesting insight into transformers.

### Weaknesses
1. The title wording could be improved for accuracy, I would suggest 'near-optimal...' instead of 'optimal...' since there is a logarithm term mismatch between lower bound and upper bound.
2. The paper can have more discussion on the separatedness assumption. For example, how realistic is the assumption?

### Questions
Following the weaknesses part, I have the following questions.

1.  What is the significance of the logarithmic gap in the paper. For instance, do authors believe this gap can be closed with further analysis, or is it likely to be fundamental?

2.1 Is there any examples of real-world datasets that satisfy or approximately satisfy this assumption?
2.2 Can authors discuss how the results might change if this assumption is relaxed?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the number of parameters and the bit complexity required by a transformer to memorize a dataset of size $N$. In the context of next token prediction, the paper proves that for any dataset with well separated tokens, there exists a transformer with $\tilde O(\sqrt{N})$ parameters+bit-complexity which interpolates it, along with a matching lower bound that $\Omega(\sqrt{N})$ parameters are necessary. In the sequence to sequence setting, the paper proves a similar upper bound with $\tilde O(\sqrt{nN})$ parameters+bit complexity where $n$ is the sequence length, and they prove a matching lower bound for hardmax transformers.

### Strengths
- The proof sketch of Theorem 4.1 is clear and well motivated
- In the next-token setting, the paper proves matching upper and lower bounds, and in the sequence to sequence setting they match for hardmax transformers
- The authors provide clear comparisons with prior work throughout the paper

### Weaknesses
- **bit complexity:** As the authors point out, the transformer needs at least $\Omega(N)$ bits, so to succeed with $\sqrt{N}$ parameters it needs a bit complexity of $\sqrt{N}$. However, this is very unrealistic as these models are trained in finite precisions, most often 16 bit precision. It is therefore unclear if these results say much about transformers ability to memorize in practice. This also affects the motivation, for example I would expected double descent (lines 102-107) to occur at $\Theta(N)$ parameters, not $\Theta(\sqrt{N})$ parameters.

- **experiments:** Related to the above point, the paper would benefit from some simple experiments that demonstrate whether such constructions are learnable or whether the empirical limit is $\Theta(N)$ or $\Theta(\sqrt{N})$ parameters. It could also be interesting to test this with various levels of precisions (FP16/32/64) to see whether higher precision really does let the transformer memorize with fewer parameters.

- minor typo on line 462: viewd -> viewed

### Questions
- If I understand correctly, neither of the setting in the paper correspond directly to the standard next-token prediction training paradigm in which the transformer needs to simultaneously predict the $i$th token given $X_{:,< i}$ for all $i$. Corollary 4.1 does not directly apply because of the varying sequence lengths, but it suggests a construction with $\tilde O(N)$ parameters+bit complexity. However, it's not clear if this is optimal since Theorem 4.3 is lower bound against the more general sequence-to-sequence task?

- Could you explain footnote 2 (line 1025)? I assume it's not possible to remove the $\log(r/\delta)$ dependence so is the goal to remove the $\log d$ or $\log C$ dependencies? If so, what is the idea?

- Is it possible to achieve the full tradeoff between bit-complexity and number of parameters? E.g. $N^{\alpha}$ parameters and $N^{1-\alpha}$ bit complexity for any $\alpha \in [1/2,1]$?

### Soundness
3

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
The paper focuses on the memorization capacity of Transformers in next-token prediction and sequence-to-sequence settings, and shows the scaling in the number of input sequences and their lengths.

### Strengths
- This paper is well-motivated and clearly presented, making it easy to follow.
- This paper is technically solid; the construction of a single-layer Transformer that can identify identical sequence ids from token-wise \((r,\delta)\)-separated sequences with near-optimal parameter order is novel. Additionally, the authors provide a parameter lower bound for the seq2seq case when the self-attention layer uses hard attention.
- Furthermore, the paper considers bit complexity, which is often overlooked in related work.

### Weaknesses
- Token-wise \((r,\delta)\)-separated sequences may not be an expressive enough model to fully capture the memorization capacity of Transformers. It appears that the authors construct a Transformer that captures contextual information by 'counting' the occurrences of tokens in a subset. However, Transformers typically capture contextual information from the mutual relationships among tokens via the attention mechanism. 
- In the proof of Theorem 4.1, the attention layer is set to uniform attention, functioning as a column-wise average operator. This over-simplification of the attention mechanism limits the ability of the work to reveal how attention contributes to memorization, as the attention mechanism should gather context-aware representations of tokens based on their relationships with others in the sequence. Therefore, how do the authors understand the role that the attention mechanism plays in the memorization capacity of the Transformer within their framework?
 - The proof for the parameter lower bound in seq2seq cases relies on hardmax activation, deviating from the real-world scenario where the attention layer adopts softmax activation. Is it possible to extend the proof to the case with softmax activation?

### Questions
See weaknesses above.

### Soundness
3

### Presentation
3

### Contribution
2
