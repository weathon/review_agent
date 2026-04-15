# Universally Optimal Watermarking Schemes for LLMs: from Theory to Practice

- Decision: Reject
- Scores: 6, 3, 3, 6

## Abstract
Large Language Models (LLMs) boosts human efficiency but also poses misuse risks, with watermarking serving as a reliable method to differentiate AI-generated content from human-created text. In this work, we propose a novel theoretical framework for watermarking LLMs. Particularly, we jointly optimize both the watermarking scheme and detector to maximize detection performance, while controlling the worst-case Type-I error and distortion in the watermarked text.  Within our framework, we characterize the _universally minimum Type-II error_, showing a fundamental trade-off between detection performance and distortion. More importantly, we identify the optimal type of detectors and watermarking schemes. Building upon our theoretical analysis, we introduce a practical, model-agnostic and computationally efficient token-level watermarking algorithm that invokes a surrogate model and the Gumbel-max trick. Empirical results on Llama-13B and Mistral-8$\times$7B demonstrate the effectiveness of our method. Furthermore, we also explore how robustness can be integrated into our theoretical framework, which provides a foundation for designing future watermarking systems with improved resilience to adversarial attacks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper finds the optimal watermarking scheme for LLM-generated texts. This is not practical therefore they proposed suboptimal but more practical schemes.

### Strengths
**S1. Awesome theoretical derivations**

**S2. Quite a readable paper despite theoretical with high technicality**

### Weaknesses
**W1. Is this watermarking?**

The main part of the submission deals with the analysis of a problem that is not watermarking per se. Therefore, it is wrong to claim "*we characterize the entire class of detectors and watermarking schemes that are universally optimal*".

The root of the problem is the non-causality of the scheme which samples jointly the text and the `auxiliary data' for detecting it. Then this data is magically transferred to the detector. This is not practical, so this is not watermarking. The auxiliary should be generated first (from the key and maybe *previous* X), and then the watermarked text X is generated.

I am referring to:
- line 182: "*We assume the auxiliary sequence can be fully recovered from X and the key*".
- line 304: "*...the detector having full access to $\zeta$*". This is not the problem. The problem is that this $\zeta$ is generated along with $X$. $\zeta$ should be sampled first, shared with the detector, then the embedder samples $X$ given $\zeta$.
- line 355: "*if the watermark sequence $\zeta$ is shared between the LLM and the detector.*"
- line 366: "*A challenge in implementing the optimal watermarking scheme is transmitting $\zeta$ to the detector*".
- line 368: "*This alternative watermarking ... results in higher minimum Type-II error*". I am afraid this is not an alternative. This is the only **watermarking** scheme that holds in practice.
- line 810: Again a problem of causality: $P_{\zeta|X}$ reads as first generate $X$, then forge the key $\zeta$ that allows the detection of $X$.

- line 256: Let me summarize the construction:
1. From $P_X$, we construct $P^*_X$,
2. With the help of function $g$, we construct $P^*_{X,\zeta}$ such that the joint distribution is not null only when $g(X)=\zeta$.
3. We sample from this joint distribution one occurrence of $X$ and $\zeta$.
4. Auxiliary $\zeta$ is magically transmitted to the detector.
5. Later on, the detector receives either the corresponding text $X$ or a random text $Y$.
There is little chance that $g(Y) = \zeta$ (ie. less than $\alpha$) whereas $g(X) = \zeta$ by construction (unless $\zeta$ is redundant).
 
I do not see the use of that scheme and I doubt that this is watermarking.
For instance, suppose that we number all the text from 1 to $|V|^T$. Function $g$ is indeed this indexation function.
Basically, the embedder samples a $X$ and tells the detector "Hey, I have generated text number $\zeta=g(X)$!". When the detector sees that text number, it shouts "Detected!". I do not see any scenario where this functionality could be useful.

However, two novel and practical **watermarking** schemes are also proposed in this paper. It is a pity that their description/benchmarking takes so little space. There is no comparison between the 2 (uniform key and SLM side-informed detector). What about the impact of the approximate distribution from the surrogate SLM. Plenty of questions left unsolved.

As for Section 6 supposedly presenting a 

**W2. Limitations**

The auxiliary data is a discrete random variable. Is this a limitation? Many schemes use a continuous rv.
There seems to be a limitation as well on the level $\alpha$ in the practical schemes which is not clearly stated. 

**W3. Experimental setup**

- line 426: Surrogate models are not really SLMs. 7B parameters, this is big. Remove "*both*".
- line 428: Given $\eta$ and $T$, $\alpha$ is a function of $\lambda$. I thought it was the other way around: given $\alpha$ and $T$, $\eta$ is a function of $\lambda$. BTW, what is the value of $\lambda$?
- line 431: There are subtle differences between EXP-edit and Gumbel-Max. EXP-edit is nothing more than Gumbel-max but with a different seeding approach (no hash window) and an enhanced decoder based on edit distance. Gumbel-Max uses a hash window. Its size is not mentioned. In practice, since no token is added or removed in the attack, the 2 schemes should share similar performances unless Gumbel-Max uses a large hash window. This is not the case here so the size of the hash window matters a lot. 
- line 443: ROC-AUC is not very informative. TPR@FPR is much better, however the FPR values (1% or even 10%) are quite large for practical application.
- line 458: "*Our watermarking method demonstrates superior performance ... This success stems from the design of our watermarking scheme*". Not only. It is also because your detector is side-informed by the SLM. Maybe the other schemes could benefit from the SLM as well. In other words, the comparison is not fair.
- line 493: "*Utilizing Alg. 1, we generate 8000 watermarked sentences*". I do not understand. You are empirically measuring the FPR, so you operate under H_0; there is no need for Alg. 1, and there is no need for watermarked sentences. This is very doubtful.

**W4. Minor comments**

- line 107: You may also cite
*Optimal Watermark Embedding and Detection Strategies Under Limited Detection Resources*, Merhav and Sabbag, IEEE Trans. Inf. Theory, Jan. 2008. *On causal and semicausal codes for joint information embedding and source coding,*, N. Merhav and E. Ordentlich,  IEEE Trans. Inform. Theory, Jan. 2006.
- line 136: It is confusing to denote both the original LLM distribution (used in hypothesis H_1) and the human-generated text (used in hypothesis H_0) by Q.  Under H_1, Q is given and the embedding crafts P. Under H_0, Q is unknown and the false alarm rate is guaranteed whatever Q (line 195).
- line 404: Union with singleton {$\{\tilde{\zeta}\}$}
- Alg.2 line 3,4: Replace $P$ by $\tilde{P}$, approximation of $P$
- line 809: $x_1^T$ are missing in the equation. Equals $\epsilon$ because the TV is defined with a factor 1/2 ... It took me ages to get it.
- line 818: (8) comes from (5)... It took me ages to get it.
- line 831, line 885: sometimes $\mathcal{S}$ is a subset of  $\mathcal{V}$, sometimes $\mathcal{S}$ is a subset of  $\mathcal{Z}$. Confusing.
- line 1040: In the special case, we are back to $\epsilon=0$, right?
- line 1056: sceheme
- line 1353: Why notation $\tilde{Q}_X$? why not $P_X$ as before?

### Questions
**Q1. Modify the first claim**
- Are you ready to modify the 1st claim to make clear that you study a problem that is simpler than watermarking, but yet interesting because it provides a lower bound on the Type II error, and some hints about the design of a real watermarking scheme?

**Q2. Efficient watermarking scheme**
- line 399, 406, 411: How can we compute $h^{-1}(\zeta)$ efficiently since a hash function is not reversible?

**Q3. Relation between parameters** 
- line 337: Definition of $\eta$. The application usually states level $\alpha$ as a requirement. How to choose $\lambda$ then?
It is unclear how $\eta$, $\alpha$, $\lambda$ are set and in which order.

**Q4. Technical detail**
- line 777: I don't get the 1st inequality. Please, explain.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
In this paper, the authors introduce a watermarking and detection framework aimed at achieving universal optimality by minimizing Type-II errors while controlling for Type-I errors and distortion. They propose a token-level watermarking scheme that is model-agnostic, utilizing a surrogate model and the Gumbel-Max trick. Their experiments on Llama-13B and Mistral-8×7B demonstrate the scheme's efficacy and robustness.

### Strengths
This paper provides detailed theoretical analyses of watermarking schemes.

### Weaknesses
1. This method is a modification based on Gumbel-Max, and employs a surrogate model during detection. However, its effectiveness may heavily depend on the similarity between the surrogate model and the original watermarked model. Notably, the authors only report results for Llama2-7B as a surrogate for Llama2-13B, and Mistral-7B for Mistral-8×7B, which are very similar pairs. This limits the algorithm’s applicability, as suitable surrogate models may not always be available.
2. During detection, the detector has no access to the user prompt $u$, which affects the reconstruction of $P_{\zeta_t | x_1^{t-1},u}(\zeta)$ and may degrade the detection performance.
3. I assume the authors apply the detection method for KGW-1 using the algorithm in lines 313–317, which is suboptimal compared to their original z-score-based detection algorithm.
4. The experiments do not report any metrics on generated text quality, and it is easy to trade text quality for improved detection ability.
5. The 1% FPR is generally too high for real-world applications. Could you provide results for TPR at much smaller FPR, such as 0.1%, 0.01% or even 0.001%?

### Questions
1. How do you determine the false positive rates? Are they calculated through a theoretical bound or by testing on unwatermarked texts?
2. Could you provide baseline results using a detector similar to the Gumbel-Max watermark detector in lines 318–320, instead of Eq. (4), to confirm the optimality of your approach?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper optimizes watermarking and detection to improve performance while managing errors and distortion. It establishes key trade-offs and identifies optimal schemes. An efficient watermarking algorithm is introduced, demonstrating effectiveness on several models and datasets. The paper also explores enhancing robustness against adversarial attacks in future systems.

### Strengths
The conclusions in the paper are well explained and proved with sufficient math notations. Furthermore, the paper introduces a optimized algorithm based on previous conclusions.

### Weaknesses
Although the proposed watermarking scheme can achieve the best performance among baselines, the numerical difference is not so significant. That is to say, for the detection accuracy evaluation, the performance of baseline method is already good enough, for both high-entropy and low-entropy texts. Then with textual attacks, although the difference is more obvious, there lacks some up-to-date robustness-enhanced algorithms such as SIR [1].
In sum, the significance of the proposed method is not sufficiently demonstrated in the paper.

1. A Semantic Invariant Robust Watermark for Large Language Models

### Questions
As explained in Weaknesses.

### Soundness
3

### Presentation
2

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
The paper considers the question: For a given amount of allowable distortion and a given false positive rate, how can we design an LLM watermarking scheme to have the lowest false negative rate?
For a certain class of watermarking schemes (which includes all existing schemes), they derive an expression for this optimal false negative rate in terms of the LLM distribution.
For the same class of schemes, they show that all optimal detectors have a simple form.
This form is not taken by most existing watermarking schemes, suggesting a possibility for improvement.

They experimentally demonstrate their ideas, showing that they are competitive with the schemes they compared against.

### Strengths
It is useful to try to understand the optimal trade-offs in text watermarking.
The typical approach is to simply suggest a scheme that addresses some perceived issue, and this work stands out in that it takes a more principled approach to the problem.

I appreciated the abstract treatment of semantic watermarks, which could be useful in the future.

### Weaknesses
It is very difficult to ascertain the extent of your optimality results. They are buried in equations and technical writing.
It would be very helpful if you described formally, but without any equations, what precisely you are claiming about optimality.
As I understand it, you prove that for watermarking schemes that fall into the formulation you describe in "Watermarking LLMs" and for fixed-length generations with no errors, you present a scheme that achieves the lowest-possible false negative rate for any given distortion level, false negative rate, and LLM generation distribution (which must be known ahead of time).
As it is, though, one has to parse a bunch of symbols and a complicated theorem (Theorem 2, which states additional results as well) to understand this.
This problem of technicality without explanation pervades the paper.

You proved the optimality of a certain form of detectors for the case of fixed-length generations, but you didn't get any optimality results for variable-length generations as far as I can tell. This is of course the practically-relevant case, so it would be better if you could say something about this case.

### Questions
- It appears that the optimal watermarking scheme you construct requires knowledge of the LLM distribution. Is this correct? If so, can the parameters of the scheme be computed efficiently (practically, or in polynomial time)?
- Can you say anything about the optimality of your detector for generations that are not of a fixed length?
- Can you describe this set of optimal detectors \Gamma^* in simpler terms? It appears that you're saying that, for any fixed watermarked text generation algorithm of the form you consider, the optimal detector simply tests whether the entire generation is exactly equal to some string computed as a function of \zeta. However, this is clearly false: If \zeta is just a collection of red/green lists for each token position, then the detector should accept many possible texts. I think that what you're trying to say is that there exists a watermarking algorithm such that this detector is optimal, but this is not what appears to be stated in Theorem 2.
- Your description of the Kuditipudi et al. watermarking scheme appears to be completely unrelated to the scheme presented in their paper. Their scheme is not hash-based at all.

### Soundness
3

### Presentation
1

### Contribution
2
