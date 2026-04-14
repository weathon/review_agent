# Fundamental Limits of Prompt Tuning Transformers: Universality, Capacity and Efficiency

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
We investigate the statistical and computational limits of prompt tuning for transformer-based foundation models.  Our key contributions are that prompt tuning on *single-head* transformers with only a *single* self-attention layer:  (i) is universal, and  (ii) supports efficient (even almost-linear time) algorithms under the Strong Exponential Time Hypothesis (SETH).  Statistically,  we prove that prompt tuning on such the simplest possible transformers are universal approximators for sequence-to-sequence Lipschitz functions.  In addition, we provide an exponential-in-$dL$ and -in-$(1/\epsilon)$ lower bound on the required soft-prompt tokens for prompt tuning to memorize any dataset with 1-layer, 1-head transformers.  Computationally, we identify a phase transition in the efficiency of prompt tuning, determined by the norm of the *soft-prompt-induced* keys and queries, and provide an upper bound criterion.  Beyond this criterion, no sub-quadratic (efficient) algorithm for prompt tuning exists under SETH.  Within this criterion,  we showcase our theory by proving the existence of almost-linear time prompt tuning inference algorithms.  These fundamental limits provide important necessary conditions for designing expressive and efficient prompt tuning methods for practitioners.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper considers the approximation power of the transformer architecture. The authors prove that single-head, single-layer attention is a uniform approximator of a certain class of Lipschitz functions over the space of sequences of tokens. The authors also provide results regarding the ability of transformers to memorize a dataset, where they improve on previous results regarding the number of fully connected layers needed to memorize. Finally, they provide a complexity-theory style result regarding the efficiency of prompt tuning. Their result indicates that prompt tuning is possible in nearly linear time under several assumptions.

### Strengths
* The paper is well-written, other from a few minor typos. There is clear effort to clearly lay out the problems and their solutions, including statements of intermediate lemmas.
* The problem clearly addresses an important problem. It's worth knowing the approximation power of the transformer architecture, given its widespread use. 
* The results also improve on the previous work of (Wang et al., 2023) in a non-trivial way.

### Weaknesses
**Organization and presentation.**
* In general, it's inarguable that this is not an easy paper to read. This is the nature of universal approximation style papers; the technical assumptions needed for this work can be daunting. However, there are a few things that I think the authors could do to make this paper appealing to more "casual" readers.
    * Rather than trying to state the theorems in full detail, opt for informal statements, with the rigorous version deferred to the appendix. This would likely lighten the load in the main text.
    * No matter how tempting, do not use \vspace{} with a negative argument. The paper is hard enough to read as is; decreasing the skips between lines does not help this whatsoever.
    * Try, whenever possible, to include equations on their own lines. There too many long-ish equations displayed in-line in this paper. As a result, I had a hard time reading this paper; I expect others will feel similarly.
* The paper could also benefit from slightly more thoughtfulness around the notation used.
    * Define notation **before** it is used. The authors use a lot of notation in the introduction that is defined in the next section. 
    * Similarly, there is confusion about the indexing variable $p$. My understanding is that this variable is used both as an indexing variable for token sequences of length or index $p$, and as the norm-type (e.g., for norms on an $L^p$ functional space). Using separate variables would improve readability.
    * Using (capital) $Z$ and (lower-case) $s$ to denote the hidden dimension is confusing. Perhaps it would be better to stick to either capital or lower-case letters when they play the same role.
   * This may just be a typo, but in Problem 2, $d_p$ takes three arguments the first time it's used, but it is defined later on (in the next sentence) as having only two arguments. I think perhaps there is a missing underscore before the $L_p$, so it ends up being $\tau([P,\cdot])_{:,L_p}$.
    * The word vocabulary is a bit overloaded. It generally is used to mean the codomain of the tokenizer. But here it's more like the set of columns of the datasets $Z^(i)$. It would be worth disambiguating this.

**Assumptions.**
* This paper would benefit from expanding on why each of its assumptions is reasonable. This is lacking throughout the paper. One could argue that this doesn't matter for universal approximation results. And yet, the fact remains that these results would be stronger and more useful if the authors made clear whether the assumptions hold in cases closer to real-world problems.  Some further comments:
    * Is the class of seq2seq Lipschitz functions a reasonable class of functions to consider? The definition under Def. 2.3 indicates that the functions are Lipschitz over the space of tokens unless I've misunderstood something. But why should we expect the space of tokens to admit local structure that makes Lipschitz meaningful? In general, tokens in the Euclidean neighborhood of other tokens may not share any semantic similarity. It makes me wonder whether there might be a better class of functions for a universality result.
    * I have a similar concern about the token separation assumption in Def. 2.5. This seems a bit unreasonable given the possibility that the dataset may contain repeated queries. It would be worth explaining how reliant the theory is on these results.

**Miscellaneous.**
* The section on computational limits (Section 3) felt relatively rushed to me. It wasn't clear why this problem is related to $k$-SAT, or what the reduction is that powered Theorem 3.2. The original limit was 10 pages, so perhaps the additional page of content could be used to expand on this part.
* The line at the end that says "By the formal nature of this work, our results do not lead to practical implementations. However, we anticipate that our findings will offer valuable insights to future prompt tuning methods" was disappointing. Not only is there no attempt to ground these ideas in real world settings, but the statement regarding the impact of this work is, in a word, vague. Is there not more that the authors can say here? This is an opportunity to say why you think that this research is impactful. And if you don't think that this paper will be impactful, it's worth asking why it was carried out in the first place. Of course, no one is expecting a paper on uniform approximation to be immediately applicable to modern LLMs. However, it would be worth speculating for at least a paragraph about why you think these results will be helpful moving forward, i.e., what kinds of "valuable insights" do you think this work will yield?

### Questions
See above

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
3

### Summary
This paper studies the theoretical properties of 1-layer 1-head transformers in terms of their approximation power and also the existence of efficient algorithms to approximate arbitrary 1-layer 1-head transformers with weights of bounded norms. There are 3 main theoretical results in the paper. The first result (theorem 2.1) shows that a single self-attention layer with sufficiently large number of MLP layers can approximate any Lipschitz function that maps one sequence of real-value vectors to another sequence of real-valued vectors. The second result (theorem 2.3) shows that a 1-layer 1-head transformers with sufficiently wide 2-layer MLPs can memorize any dataset generated by Lipschitz functions with prompt tuning with sufficiently long prompt. Finally, the third result (Theorem 3.1) shows that the Prompt Tuning Inference problem can be approximated efficiently in nearly linear time (in terms of the sequence length) through a reduction to a previously known result (AttC).

### Strengths
The paper is technically strong. It provides several improvements over previous results on the topic.
While I did not verify every single proof, the overall argument seems sound and the conclusion plausible.
The paper is fairly well-written for such a technically involved paper, and I believe the results in this paper will be useful for future work studying the approximation properties of transformers.

### Weaknesses
Overall, while the paper is technically solid, I have a hard time seeing it being impactful for communities beyond those studying the representational power of transformers. 

I consider myself to be reasonably well-informed about the development of deep learning theory. These kinds of approximation results are certainly nice to know, but they rarely make any impact in practice or even inform any algorithmic design. For example, the authors claim that:

> These fundamental limits provide important necessary conditions for designing expressive and efficient prompt tuning methods for practitioners.

I do not see how these results will have any impact on how we do prompt tuning right now. What does it tell us that we do not already know? If there are, I would like to see them discussed in more detail in the paper. For example, does this result says about why prompt tuning fails in certain scenarios?

I would also like to see related work in the main body of the paper and a more detailed discussion of how this result relates to prior works. The page limit has been raised to 10 so there is no reason to leave them in the appendix.

### Questions
> We study the fundamental limits of prompt tuning transformer-based pre-trained models (i.e., foundation models) in two aspects

Where does the pretraining affect this analysis? As far as I can tell, the paper only considers the representational power of the transformer. Why does it matter if the transformer is pre-trained or not? If pretraining matters in the analysis, I would expect to see some assumptions about the representation from pretraining but I am not sure if there is any.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the fundamental limits of prompt tuning and derives several interesting theoretical properties on single-head transformers with a single self-attention layer. The authors 1) show that prompt tuning can universally approximate any sequence-to-sequence function, subject to some continuity assumptions,  2) show that prompt tuning can memorize arbitrary datasets and 3) prove the existence of almost-linear time prompt tuning algorithms, which builds the theoretical foundation on the computational time required for prompt tuning.

### Strengths
Overall, given the theoretical nature of the paper, it is a dense read by design, but the good presentation has helped a lot by clearly outlining the different contributions (e.g., universality results, memorization properties and the computational time analysis). The paper is quite solid in terms of the theoretical contributions. The contributions are outlined cleanly and the flow is great. The theoretical results also represent a significant contribution over existing works, such as [1].

References

[1] Yihan Wang, Jatin Chauhan, Wei Wang, and Cho-Jui Hsieh. Universality and limitations of prompt tuning. Advances in Neural Information Processing Systems (NeurIPS), 36, 2023a.

### Weaknesses
To begin with my feedback, I'd like to mention that while I have a grasp about the high-level ideas and contributions of the paper, I have not checked the math and derivations in detail although they seem reasonable; I am not an expert in this area, so I might not have sufficient knowledge in judging the merits and significance of the work w.r.t. the previous works, so I defer the assessment of these parts to other reviewers and the AC. 

I understand the paper outlines many "there exists" type of results which cannot be verified easily with practical algorithms, however, I do think the authors could give add some discussions on in what circumstances their theoretical results could lead to practical implications, especially given that prompt tuning itself is a very *practical* low-cost adaptation technique, so for the results to be relevant and of interest beyond the theoretical community, some kind of discussions about practical impact are essential. There also seems to be some contradictions given the paper suggests that even a shallow, single-head transformer is already universal yet there is some empirical evidence that prompt tuning is more effective with deeper transformers -- while I understand some gap between theory and practice can be acceptable especially given that many theoretical properties of prompt tuning are not well-known thus far, I wonder whether the authors could give any comments on why that is the case? Is it possible that there are unrealistic assumptions in the theoretical analysis?

### Questions
Please refer to "Weaknesses" above.

### Soundness
3

### Presentation
3

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
The paper investigates the limits of prompt tuning in transformer-based models by focusing on the simplest architectures with a single self-attention layer and head. The authors prove that such simple transformers can universally approximate any Lipschitz continuous sequence-to-sequence function through prompt tuning. They also demonstrate that these transformers can memorize any dataset via prompt tuning by establishing an exponential lower bound on the required number of soft-prompt tokens based on data dimension, sequence length, and desired error tolerance. Additionally, they identify a computational efficiency phase transition determined by the norms of the soft-prompt-induced keys and queries. They establish an efficiency threshold beyond which no sub-quadratic algorithm exists under the SETH, but within this threshold, they prove the existence of almost linear-time inference algorithms for prompt tuning.

### Strengths
1. The paper advances the theory of prompt tuning by proving that even the simplest transformer architectures—with a single attention head and a single self-attention layer—can universally approximate any Lipschitz continuous sequence-to-sequence function.
 
2. The paper notably discovers a phase transition in computational efficiency based on the norms of the soft-prompt-induced keys and queries. It provides a criterion under which prompt tuning can be performed in sub-quadratic time and demonstrates the existence of almost linear time algorithms under certain conditions.

3. The paper introduces new applications, such as the chained reduction of piece-wise constant approximations, to prove universality results. It also extends the contextual mapping property to any-rank weight matrices in single-layer attention mechanisms, improving the rank-1 limitation of previous work.

### Weaknesses
1. The paper contains numerous grammatical errors, and some sentences are poorly constructed. A thorough proofreading is required to enhance the readability of the text.

2. While the paper brings new applications of previous concepts to function approximation, the technical novelty appears limited, with many proofs in Section 2 overlapping significantly with those used in "Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators?" and Universality and Limitations of Prompt Tuning" papers.

3. The paper does not include experiments to showcase any of the theoretical results. The authors could demonstrate the approximation capabilities of single-head, single-layer transformers for sequence-to-sequence functions in Section 2. The authors can even use synthetic datasets designed to have known Lipschitz properties. Deeper transformers with more heads and layers might be used for comparison.
 
4. Another empirical comparison might be examining the relationship between the number of soft-prompt tokens and the model's ability to memorize datasets, as suggested by the exponential lower bound. Again, one might utilize synthetic datasets where the complexity (e.g., vocabulary size, sequence length) can be controlled.

5. I think the authors should perhaps discuss and more on the results and analysis of the “Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators?” and compare the results as they prove that transformers with one self-attention layer are universal approximators for permutation equivariant continuous functions. Also, explaining how the results in the current paper relate to or differ from theirs might provide insights.

### Questions
1. The paper "Are Transformers with One Layer Self-Attention Using Low-Rank Weight Matrices Universal Approximators?" stated that they easily extended their analysis of contextual mapping to masked self-attention, which is what's actually used in practice. Since such an extension might be beneficial for completeness, could the authors elaborate on why they haven't addressed it? Was it more challenging in this case?

2. The identified efficiency criterion (Theorem 3.1) requires controlling the norms of the soft-prompt-induced keys and queries to be $O(\log(L_p + L))$. In practical applications, particularly with large-scale models and datasets, how feasible is it to meet this criterion?

**Minor Remarks**

1. In order to make them multiplication-compatible, $W_O^i$ should be $d \times s$, and this doesn’t propagate to later sections.

2. The infinity norm is already defined in matrices, which gives the maximum row sum, and perhaps a better notation would be $||M||_{max}$ for clarity.

### Soundness
2

### Presentation
1

### Contribution
2
