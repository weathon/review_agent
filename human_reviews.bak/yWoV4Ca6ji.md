# Towards Understanding the Universality of Transformers for Next-Token Prediction

- Decision: Accept (Poster)
- Scores: 5, 6, 6, 3, 6

## Abstract
Causal Transformers are trained to predict the next token for a given context. While it is widely accepted that self-attention is crucial for encoding the causal structure of sequences, the precise underlying mechanism behind this in-context autoregressive learning ability remains unclear. In this paper, we take a step towards understanding this phenomenon by studying the approximation ability of Transformers for next-token prediction. Specifically, we explore the capacity of causal Transformers to predict the next token $x_{t+1}$ given an autoregressive sequence $(x_1, \dots, x_t)$ as a prompt, where $ x_{t+1} = f(x_t) $, and $ f $ is a context-dependent function that varies with each sequence.
On the theoretical side, we focus on specific instances, namely when $ f $ is linear or when $ (x_t)$ is periodic. We explicitly construct a Transformer (with linear, exponential, or softmax attention) that learns the mapping $f$ in-context through a causal kernel descent method. The causal kernel descent method we propose provably estimates $x_{t+1} $ based solely on past and current observations $ (x_1, \dots, x_t) $, with connections to the Kaczmarz algorithm in Hilbert spaces. We present experimental results that validate our theoretical findings and suggest their applicability to more general mappings $f$.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper aims to advance our understanding of causal Transformers for next-token prediction in autoregressive settings by showing how to construct networks that can solve the next-token prediction in-context learning task.

### Strengths
**Despite not being very familiar with kernel methods and learning theory (but somewhat familiar with Transformers and in-context learning), I believe that the authors' contributions are relevant and interesting, though I cannot assess their (learning theoretical) details.**

Particular, I find the paper is strong in the following aspects:
- crisp and clear setup in the abstract
- intuitive Fig 1 (for the setting, not for the results)
- very well summarized contributions list
- the results seem to be strong (I am uncertain about how realistic the setting is)

Nonetheless, as I will detail below, **I think the presentation needs to be significantly improved**.

### Weaknesses
### Major points
- the paper is very dense, even though the authors could have used an additional 10th page. I'd suggest more explanations for the result, and less formulas in the main text
- the intuition seems to be missing (for me) as to why you need the augmented tokens. What do they mean?
- Figure 2 needs to be explained; the one-sentence reference in L342 does not explain (for me, at least) what all the quantities in the figure are and how I should think about them).
- **It is unclear to me how a Transformer can implement Eq (7) if one problem is with the non-causal descent.** You still need to modify the training method, right? Please elaborate how this works/correct me if I am wrong (I see how you can construct the Transformer, but I do not see how the Transformer itself can account for the change in the descent method). If this point is about the "meta-optimization" during solving the in-context learning task, then please say so explicitly
- What is the rationale behind the construction of $f$ in the _"More Complex Iterations"_ paragraph?

### Minor points
- please use equation numbers for the main equations 
- please define all quantities before you use them
- Definition 1: what is $S^{d-1}$?
- Definition 2: what is $O(\cdot)$?
- Definitions 1 and 2 would be better suited as assumptions
- L206: please explain what $e$ is, what the indices stand for, and why you add multiple beginning-of-sequence tokens and tokens of "1".
- Proposition 1: what is $\mathcal{N}?$
- L246: what makes these constructions explicit? As far as I understand, you specify the dimensions, but that leaves many degrees of freedom (ie, all the elements can be of any value)?
	- From Prop. 5 and checking A.8, I get a sense what these matrices would be. Thus, as you have a contruction, please refer to this fact to the reader, otherwise, calling a matrix explicit without saying that you can calculate all the values can be misleading.
- L286: do you mean the update of $u_{t}$ depends on $x_{1:t}?$

### Questions
- L178: why is your setup better reflective of how LLMs are trained? Please elaborate.
- L194: Why does the unit-norm of $x$ imply that the kernel values for $k(x_{i},x_{i})$ are same for all $i$?
- Eq (3): why do you need to model the projection (in this exact way of picking the last $d$ coordinates)?

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
This paper theoretically studies the universality of Transformers for next-token prediction. The authors specifically consider the sequences of the form $x_{t+1} = f(x_t)$. In particular, they theoretically analyze two types of instances:  for linear $f$ and periodic sequences (exponential kernel). They construct augmented tokens and show that an explicitly constructed Transformer can learn in context to accurately predict the next token asymptotically, through a causal kernel descent method.

### Strengths
1. Very mathematically sound and interesting. The universality of this next-token predictability is a strong result. 

2. Causal kernel descent is a very intriguing framework to analyze sequence models in context prediction, taking the important aspect of masking out the future into account.

### Weaknesses
1. The theoretical setting is limited to when $f$ is linear and when the sequence is periodic, with each point in the context being on the unit sphere. Neither cases seem too complicated in the first place, and so the "universality" does not seem well justified in the context of Theorem 1. Perhaps the authors can justify the reduction to these sub-classes of functions more explicitly.

2. Both the process of generating the augmented token $e_t^0$ from $x_{0:t}$, as well as the causal kernel descent iteration theoretically involves an infinitely deep Transformer as $n\to\infty$. It might be more informative to discuss the theoretical implications of finite depth. 

3. The universality of next token predictability result is asymptotic in the context length $t$. Although the convergence is fast enough (exponential), most empirical observations are in the non-asymptotic regime in terms of context length. It might be good to explicitly provide analysis on this.

### Questions
1. 250 typo $\epsilon_1(n,t) = \mathcal{M}^n(x_{1:t}) - \mathcal{M}(x:1,t)$?

2. In the experiment section, how did you compare fine-tuning $\mathcal{M}^n$ with infinitely deep model? Did you just take $\mathcal{M}$ to be the ground truth? (Figure 5 is missing)

### Soundness
3

### Presentation
3

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
The paper studies the approximation abilities of causal transformers when predicting the next token in-context for autoregressive sequences. In particular, it focuses on specific instances where the context-dependent map $f$ determining the next token $x_{t+1} = f(x_t)$ in the sequence belongs to the RKHS of a given kernel $k$ and is either linear or the sequence is periodic.  The authors introduce the causal kernel descent method and theoretically analyze its convergence properties. They provide a construction of the transformers layers with linear, exponential and softamx attention that can implement this method.

### Strengths
- The paper is well-written. 
- The problem is conceptually well-motivated.
- The proposed method and analysis are clearly explained with all necessary details.
- The manuscript includes both rigorous proofs and empirical experiments that validate the findings.
- The proof technique and its connection to the Kaczmarz algorithm in Theorem 3 are insightful.

### Weaknesses
1. The choice of normalization in the attention appears to be specifically tailored to match the kernel used in the data-generating process (see Definition 2 and Theorem 1). This design decision raises questions about the generalizability of the findings, making it unclear how these results might extend to broader settings.
2. Since the analysis is confined to the class of functions determined by the chosen kernel, it is not clear how the results illustrated in the paper contribute towards understanding the universality of transformer models.
4. The theoretical results presented in the paper rely on taking the limits as the context length $t$ and the number of layers $n$ approach infinity. However,  the empirical experiments reported in Figure 4, demonstrate that Transformers with a finite number of layers, trained using the Adam optimizer, can achieve better performance than the proposed infinite-layer model. This discrepancy suggests that finite-sized trained Transformers may implement a different algorithm than the causal kernel descent method.
5. Exponential convergence for the case of exponential kernel and exponential or softamax attention requires periodic sequences.

### Questions
1. Could the authors clarify how their results using particular kernel choices and corresponding attention mechanisms contribute to understanding the universality of transformers in learning sequence-to-sequence functions and what they mean by universality in this context?
3. In Definition 2 could the authors clarify how is the periodicity enforced in instance (3). 
4. In Proposition 1, it’s unclear what $n$ refers to, as elsewhere it denotes the number of layers, but here it relates only to the first layer that computes augmented tokens. In Appendix A.1, $n$ seems tied to the specific positional encoding. Could the authors clarify this?
5. Is it correct that in the presented model, the first layer computing the augmented tokens always uses softmax attention and does not have a skip connection, whereas the identical layers can employ different normalizations (softmax, linear, exponential) and include skip connections?
6.  In the paragraph: Augmented tokens,  could the authors clarify why $e_1^0 := (0_d,1,x_t,1,0_d,0_d)$ rather than $(0_d,1,0_d,1,x_t,0_d)$ based on the feed-forward map in appendix A.1.
7. (Minor) In Remark 1, there may be a typo in the definition of $\epsilon_1$.
8. In Equation 8 could the authors provide more insights regarding the interpretation of $\mu$ and its relation with the data-generating process?
9. For instance (4) described in section 5, did the authors try to train a model, and if so, what were the results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper considers the capacity of causal Transformers to the prediction of next token $x_{t+1}$ given autoregressive sequence $(x_1,\ldots,x_{t})$ as a prompt where $x_{t+1}=f(x_t)$. They explicitly construct a Transformer that learns the mapping $f$ in-context through a causal kernel descent method (which connects to the Kaczmarz algorithm) and prove long scope guarantees.

### Strengths
The strength comes mainly from the problem setup of studying next-token prediction, which seems to be quite an important task. The proof looks sound. The presentation of the work is clear. The authors perform empirical evaluation for their theoretical results. The discussion of theoretical results are provided in the work.

### Weaknesses
The major concern over this work is listed as follows.

(1) The problem setup is not realistic enough in the sequential relationship. In particular, the reviewer does not believe either the language or the time series admit simple relationship of $x_{t+1}=f(x_{t})$. Hence, the reviewer cannot understand how this work contributes to the understanding of Transformers on these modern ML tasks, which is believed to be very crucial to the ML community. Moreover, the reviewer believes that RNN might be able to do the same task as the results claimed in this work for Transformers. If this is true, the results might universally hold for many sequential models. Then, the unique advantage of Transformer is not highlighted.

(2) The problem setup is not realistic enough in the deterministic assumptions. The reviewer believes that either Markov chains or some random autoregressive models are necessary for the purpose of studying the universality of next token prediction of Transformers. At the current stage, this fully deterministic dynamic system seems rather naive.

(3) The limitations of this work is not properly discussed at the conclusion part.

### Questions
The questions and suggestions are given by the weakness section. The reviewer believes that this work requires significant improvement to reach his/her expectation and doesn't seem to be possibly addressed within the time frame of ICLR 2025 rebuttal phase.

However, if the author is able to resolve at least (1) or (2) in the question section, the reviewer might be able to improve his/her opinion.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends a line of work showing what transformers are capable of learning. In particular, it shows that causal transformers are capable of learning kernelized linear functions, in context, which means that each token is the same linear function of the previous token. Previous work had focused on linear attention whereas this work extends to softmax attention. It also handles period sequences.

### Strengths
This paper furthers our understanding of why transformers, like the amazing LLMs, are able to learn in context. It extends to learning linear functions with softmax attention. It provides nontrivial, rigorous guarantees as well as experimental evidence corroborating the theory.

### Weaknesses
This is not a weakness of the work as much as a fit for the conference. This work seems important but also could arguably be more appropriate for a specialized theoretical audience like various theory conferences (COLT/ALT/STOC/FOCS). Previous works have covered linear functions and the extension to kernels and softmax attention is undoubtedly important for our understanding, but the question is how many of the conference attendees will appreciate these contributions. Of course, theory is a topic on the CFP and transformers are a key interest for ICLR so it's not a bad fit.

### Questions
Just to make sure I understand, is the main technical piece here handling the softmax activation? If the goal was to extend linear activation to the kernelized setting, would that just be adding the "kernel trick" to Sander et al (2024)?

### Soundness
3

### Presentation
4

### Contribution
3
