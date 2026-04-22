# The Counting Power of Transformers

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 8, 4, 8, 4, 8

## Abstract
Counting properties (e.g. determining whether certain tokens occur more
   than other tokens in a given input text) have played a significant role in 
    the study of expressiveness of transformers. In this paper, we provide a 
    formal 
    framework for investigating the counting power of transformers. We argue 
    that all existing results demonstrate transformers' expressivity only for 
    (semi-)linear counting properties, i.e., which are expressible as a 
    boolean combination of linear inequalities. 
    Our main result is that transformers can express counting properties that
    are highly nonlinear. More precisely, we prove that transformers can
    capture all semialgebraic counting properties, i.e., expressible as 
    a boolean combination of arbitrary multivariate polynomials (of any degree).
    Among others, these generalize the counting properties that
    can be captured by C-RASP softmax transformers, which capture only
    linear counting properties.
    To complement this result, we exhibit a natural subclass of (softmax) 
    transformers that completely characterizes semialgebraic counting 
    properties. 
    Through connections with the
    Hilbert's tenth problem, this expressivity of transformers also 
    yields a new undecidability result for analyzing an extremely simple 
    transformer model -- surprisingly with neither positional encodings 
    (i.e. NoPE-transformers) nor masking.
    We also experimentally validate trainability of such counting
    properties.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This work analyzes the ability of Transformer based models to performing higher order forms of counting (e.g. counting if the product of three variables is larger than some threshold). The authors analyze the counting abilities of Transformers under various assumptions about the self-attention layer and analyze how the counting abilities of models with different self-attention assumptions relate to one another in terms of expressivity.

The main theoretical result of the paper shows that softmax attention transformers can express any counting property which can be expressed as a boolean combination of multivariate polynomial inequalities. For AHAT, the authors provide an exact characterization showing that the set of counting languages accepted by AHAT is equivalent to the set of semi-algebraic languages.

The authors complement this with empirical results showing that models can easily learn to perform classification on the set of languages of the form $|w|_b \leq |w|_a^k$ for k from 1 to 5.

I think the claims made by the paper are very interesting and meaningfully advance the field of Transformer expressivity. Most of my feedback is minor details which I think could help improve the exposition of the theoretical results However, I quite like this paper and thus have given it an 8.

### Strengths
- The main claims are interesting and novel. Showing that softmax attention transformers can recognize semi-algebraic counting languages and giving characterization of the subclass meaningfully advances our theoretical understanding of Transformer counting power
- Paper is well written. Motivation of the theoretical problem considered is given. In the sections exposing the framework and the theoretical results, examples are given which is nice. 
- At a high level, the theoretical approach taken is clean and convincing. The authors also show an interesting inexpressibility result for PARITY as well as undecidability/universality Theorems (1.3 and 1.4). The argument for two layer networks is also appealing.

### Weaknesses
* The main weakness of the paper for me is the exposition. I feel many things could be made clearer or more rigorous in the proofs/in the formalization of the problem. There are many occurrences of variables which are  referred to in text or in equations without properly being defined. I think this hinders the readability of the paper.
* I also think the experiments are quite limited. It would be interested to see evaluation on languages that are not of the form of $L_k$.  I also wish the authors would have tested on languages with different vocabulary sizes to see how this affects performance. However, the theoretical contribution is, in my opinion, strong enough that this is not a major shortcoming to the paper's contribution.

### Questions
**General Questions**

- Do we know if, in terms of counting, there are languages in SMAT but **not** in SemiAlg?
- Could the authors give somewhere a more formal/rigorous definition of the sets AHAT, AHAT[U] and SMAT in the context of semi-algebraic language recognition? No formal definition of these sets are given, but many of the main results hinge on inclusions between these and SemiAlg.
- Could you give a more precise theorem statement for Thm 1.1? Which class of Transformers are concerned by this?
- You do not use multi-head attention in your definition of the Transformer. Is there a reason for this?
- I would like for asymptotics (in terms of number of layers/width/number of heads) to be clearly stated either in or around the definition of the theorems. Currently, the only place where number of layers is referred to is in the paragraph discussing the reduction to two layers argument. I found this to be late in the text to introduce these.

**Minor Comments/Clarification Questions**

- For the proof of Prop 3.1"(each consisting of one uniform layer and several ReLU layers)" How many RELU layers? At least state big O.
- The argument for PARITY in the appendix should be introduced in a proof environment with an associated Proposition/Lemma/Theorem.
- What is $d$ line~ 633?
- What are PI and RE languages ~line 382?
- The titles for the columns/lines in Figure 1 could have clearer names.

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper theoretically investigates the capabilities of encoder-only Transformers to recognize formal languages. Specifically, it focuses on languages characterized by polynomial inequalities based on occurrence counts. The authors leverage this analysis to present a related undecidability result for Transformers. The theoretical findings are empirically validated on the language class $L_k = \{ w\in \{a,b\}^+: |w|_b \leq |w|_a^k \}$.

### Strengths
The paper's primary contribution is extending prior work on the counting abilities of Transformers. By generalizing the analysis from linear functions of occurrence counts to polynomial functions, the work advances our theoretical understanding of Transformer capabilities in this domain.

### Weaknesses
1. Given the theoretical nature of the paper and the numerous formal language concepts, the presentation would be significantly clarified by adding a Venn diagram. This could visually situate the language classes considered in this work relative to each other and to prior art, making the precise scope of the contribution more apparent.

2. The analysis is confined to encoder-only Transformers. While this is a valid methodological choice, the paper would have broader impact if it discussed the implications of these findings or extended them to autoregressive models, which are prevalent in modern applications.

3. Theorem 1.1 establishes the expressiveness of Transformers for semialgebraic counting properties. However, this claim would be much stronger with a corresponding analysis of the required model size (e.g., depth, width, or number of heads) as a function of the complexity of the counting problem. For instance, do the required Transformers scale polynomially or exponentially with the degree of the polynomial inequalities?

4. The empirical validation of this paper is limited to the single language class $L_k$. The paper's value could be enhanced by either broadening the experimental scope to other complex formal languages or by discussing the potential implications of these counting properties for more conventional NLP tasks, even at a high level.

### Questions
1. Could the authors elaborate on the size requirements (e.g., depth, width) for the Transformers needed to express the counting properties discussed in Theorem 1.1 and other results? Specifically, how does the required model size scale with the parameters of the formal language (e.g., the degree of the polynomial)?

2. The current experiments are focused on $L_k$. Are there other language classes the authors considered that would serve as interesting and challenging testbeds for this theory?

3. To help bridge the gap between this theory and practice, could the authors speculate on any practical NLP tasks (e.g., in semantic parsing, logical reasoning, or program synthesis) where these specific polynomial counting abilities might be relevant?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper provides a theoretical analysis on transformers’ ability to perform counting operations. It is shown that transformers can capture all semialgebraic properties, meaning they can evaluate polynomials where variables are characterized by symbol counts (for instance, accepting strings s.t (#a)^2 + (#b)^3 > 0). The proof mainly consists of demonstrating that average-hard attention transformers (AHATs) with no positional embeddings exactly characterize semialgebraic sets. As a direct corollary, they are able to show that AHATs with no PEs cannot recognize Parity. Furthermore, by connecting this result with the MRDP theorem (which states that there is no algorithm that determines whether some given Diophantine has a solution), they provide an undecidability result for transformers: it is undecidable to determine whether their language is empty or not.

### Strengths
1. The perspective on semialgebraic sets rather than semilinear sets is novel and generalizes prior results on the counting power of transformers
2. The corollary on inexpressibility of Parity is interesting and accompanies a rich body of work tackling this question
Novel tools to me such as semialgebraic sets and Parikh images were introduced well enough for me to understand the technical parts of the paper
3. Introduction is well written and puts in perspective previous work on semilinear counting with transformers

### Weaknesses
1. Importantly, this paper disregards the impact of precision. In the finite-precision regime (which describes transformers used in practice), it is impossible to store counts from uniform attention for any input string. Recently, the expressive power of fixed-precision transformers (SMATs and AHATs) has already been characterized by a subclass of regular languages [Li and Cotterell, 2025] (and therefore can not perform counts across all possible strings), undermining the relevance of the paper’s main claim on counting. I would recommend at least mentioning this inherent limitation.
2. The experiments consist of training transformers on a single type of polynomial (comparing (#b) and (#a)^k) rather than a more diverse set of polynomials with different coefficients of different degrees. Experiments on polynomials with a larger alphabet, different coefficients with different degrees would consolidate the claims of the paper. Even better, training on randomly sampled polynomials would be a great contribution.

### Questions
N/A

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the counting power of Transformers.
While prior work on Transformer expressiveness has shown that Transformers possess semilinear counting properties, this paper demonstrates that Transformers can express all semialgebraic counting properties.
It also shows that Average Hard Attention Transformers (AHAT) without positional encodings (PEs), as well as their subset AHAT[U] that uses only uniform layers, precisely capture semialgebraic counting properties.

### Strengths
- Understanding the expressive power of Transformers is an important research topic.  
  This paper presents new results on counting capabilities that were not clarified in previous work.
- The paper is concise and readable.
- The theoretical results are supported by experiments.
- Although I had some questions, the theoretical part appears mostly correct.

### Weaknesses
- It is unclear how practical it is to apply Transformers to nonlinear counting problems as discussed in this paper.  
  For sequences such as text, which Transformers typically handle, input order is important, and tasks are generally not permutation-invariant.  
  Therefore, studying permutation-invariant input properties may have limited practical relevance.  
  As the authors discuss in the paragraph beginning at Line 263, combining counting properties with other characteristics is interesting.  
  However, it is not clear whether nonlinear counting is necessary in such cases.  
  In fact, the use case shown at Line 264 can be realized with linear counting.

- The proof of the main result, Proposition 3.1, seems somewhat trivial.  
  The idea in Step I, computing the frequency via a Transformer layer, has already appeared in prior work such as Yang & Chiang (2024).  
  Step II, which performs multiplication, also appears straightforward.


## Minor comments

- Line 22: properties.. => properties.
- Line 161: (a.k.a. Parikh map => missing ')'
- Line 180: I think the fourth vector should be (0,0,1, 1/4) if we use the one-hot embeddings.
- Line 190: (x_1, \ldots, x_n) => (x_1, \ldots, x_m)? 
- Line 205: duplicated ':='.
- Line 213: x_j \tau => x_j / \tau
- Line 227: duplicated ','.

### Questions
- Are there practical situations where nonlinear counting is required for tasks usually solved with Transformers?
- Is it possible to define a new class of language, e.g., LTL with nonlinear counting?
- I could not understand the paragraph beginning at Line 355.  
  Regarding the arbitrary choices among the $2^{r+(m+1)^2a}$ options representable by AHAT, why is it “easy to do” to construct the polynomial expression of $u\_{\ell, i}$?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper formally studies the counting power of transformers, i.e., which counting properties of the input can transformer models represent. The authors substantially expand on prior understanding, showing that the answer goes well-beyond previously studied (Boolean combinations of) linear properties, to (Boolean combinations of) polynomial properties of the number of occurrences of various tokens in the input. Along the way, they also define and motivate counting properties, and provide a small empirical confirmation of the findings. Towards the end, they make connections to universality and undecidability of certain classes of transformers.

### Strengths
* Well-motivated and nicely written paper. While technically strong, its pitch and claims will also be accessible to broader audience not deep into theoretical research.

* I liked the structure of the paper and how it (quickly) conveys the findings and uses many examples to explain concepts.

* That the studied transformer model uses simple (or no) position encodings and the standard softmax attention (either directly or through AHAT[U] which is a special case of softmax) is a positive, especially compared to some efforts that have relied to unrealistically complex position embeddings and other design choices in their analysis.

* The main result about transformers being able to capture counting properties well beyond (combinations of) linear properties is a strong technical advance. Having an empirical support for it is a plus.

* The connections to undecidability are intriguing, though I must say the corresponding sections can use some more clarity of exposition.

### Weaknesses
* While it's valuable to have an **empirical validation**, I felt that section 6 is not as well described and discussed as the rest of the paper. E.g., even the metrics mentioned in the caption of Fig 1 need clarification.

* There are some places where it would be valuable to state which **design decisions / assumptions / choices** play an important role. E.g., it seems to me that Prop 4.1 (that whatever NoPE-AHAT can compute is expressible semi-algebraically) relies on the assumption of ReLU as the non-linearity; I suspect it will also work with any polynomial non-linearity, but not with sigmoid, inverse-tangent, or other choices that have been used in practice. (And this is fine, I just think it's better if the authors can call it out.)

* The paper can also use a clearer treatment of the **datatype** assumed for the transformer model. Section 2.1 starts with *real* vectors, which clearly isn't realistic, at least for arbitrary precision. Later, the ReLU paragraph switches to *rationals*, leaving some lack of clarity. The choice of datatype -- and importantly the *precision* (how many bits are allocated, and whether they depend on the input length $n$ -- is an important consideration in transformer expressivity results, but seems to be lacking here. E.g., when, in the proof of Prop 3.1 (that semi-algebraic counts can be expressed by NoPE-AHAT[U]) the transformer is computing $u_p[j] \in [0,1]$, this is presumably expressed as a rational where the numerator (and denominator) can grow very quickly. Their size should be bounded in terms of $\ell$ and $n$, though. It would be helpful to get some clarity on this front.

* Some of the proof ideas can use a short discussion of the **intuition**. E.g., in the proof of Prop 3.1, we are multiplying two integers (represented as rationals). A priori, it's unclear how a transformer might be able to do so! The key observation, I believe, is that one of the items being multiplied, namely $x_i$, is always a *count* of letters in the input word, and thus distributed across the input as captured by $u_p[i]$ being 0 or 1 at various positions $p$. Thus, one can get around the usual difficulty of multiplication by instead multiplying $y_i$ with a 0 or a 1 at each position, ensuring that there are exactly $x_i$ positions with a 1, and then using attention to aggregate. Is this the right intuition? In any case, adding some intuition would help.

### Questions
Please see the comments above about the (implicit) assumptions on the datatype and precision, and on intuitions. Clarity on any of these fronts would be great to have!

### Soundness
3

### Presentation
4

### Contribution
4
