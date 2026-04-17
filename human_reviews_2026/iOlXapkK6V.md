# Why Can't Transformers Learn Multiplication? Reverse-Engineering Implicit Chain-of-Thought Reveals Challenges of Learning Long-Range Dependencies

- Decision: Reject
- Scores: 10, 2, 4, 4

## Abstract
Language models are increasingly capable, yet still fail at a seemingly simple task
of multi-digit multiplication. In this work, we study why, by reverse-engineering a model that successfully learns multiplication via *implicit chain-of-thought*, and report three findings: (1) Evidence of long-range structure: Logit attributions and linear probes indicate that the model encodes the necessary long-range dependencies for multi-digit multiplication. (2) Mechanism: the model encodes long-range dependencies using attention to construct a directed acyclic graph to "cache" and "retrieve" pairwise partial products. (3) Geometry: the model implements partial products in attention heads by forming Minkowski sums between pairs of digits, and digits are represented using a Fourier basis, both of which are intuitive and efficient representations that the standard fine-tuning model lacks. With these insights, we revisit the learning dynamics of standard fine-tuning and find that the model converges to a local optimum that lacks the required long-range dependencies. We further validate this understanding by introducing an auxiliary loss that predicts the ``running sum'' via a linear regression probe, which provides an inductive bias that enables the model to successfully learn multi-digit multiplication. In summary, by reverse-engineering the mechanisms of an implicit chain-of-thought model we uncover a pitfall for learning long-range dependencies in Transformers and provide an example of how the correct inductive bias can address this issue.

## Human Reviews

## Human Reviewer 1

### Rating
10

### Rating Number
10

### Confidence
4

### Summary
This paper investigates why Transformers fail at multi-digit multiplication. The authors reverse-engineer an implicit chain-of-thought (ICoT) model that successfully learns 4×4 multiplication and compare it to a standard fine-tuned model that fails. Through mechanistic analysis, they discover that the ICoT model encodes necessary long-range dependencies by constructing a binary-tree-like attention pattern that caches pairwise partial products across timesteps for later retrieval. The model also uses sophisticated geometric representations, encoding digits with Fourier bases that form a pentagonal prism structure.

In contrast, standard fine-tuning gets stuck in a local optimum where only the first, last, and some early digits are learned correctly - the model fails to learn the middle digits that require tracking complex dependencies. To validate their understanding, the authors introduce an auxiliary loss that supervises the model to predict intermediate "running sums," providing the inductive bias needed to achieve 99% accuracy without chain-of-thought supervision. This work reveals a fundamental challenge in how Transformers learn long-range dependencies under standard training and demonstrates that while task-specific fixes exist, more general solutions are needed.

### Strengths
This is an excellent paper. 

The mechanistic investigation of the ICOT model builds up over several layers, from theoretical analysis of long-range dependencies, demonstrated through logit attributions, linear regression probing, use of some attention heads to handle long-range dependencies, the outputs of some attention heads forming a Minkowski sum, a 3D two-level PCA analysis, and finally a Fourier analysis. 

The pentagram prism structure, revealed in the Fourier analysis, is a novel insight into how the models encodes information about our base 10 counting system. This “tool-like” structure appears to embed: A) 10 is divisible by 5 and 2 B) concept of odd and even numbers C) potentially a way to calculate n+5 and n-5 (between layers) and D) n+4 and n-4 (around rings).

The Figures are excellent - particularly 1 & 4 as they represent hard-to-diagram concepts. Figure 6 is beautifully presented - the pentagon prism gives a really great insight into how this model represents base 10 digits in model space.

The ICOT model insights lead directly to a modified loss function allowing SFT of a model to perform multiplication accurately. This strengthens the case for the ICOT insights being valid.

### Weaknesses
None to speak of. I think the Conclusion undersells what the paper achieves. 

Minor points: 
- Minkowski sums are first used without explanation. Consider adding “(adding sets of vectors geometrically)”.
- Section 3.3, the notation “ATT superscript 2 subscript 3” is not explained before use, and differs from the “Layer 2 Head 3” notation used in Figure 4. Later “ATT superscript 1 (i, j)” is used with little introduction. Consider standardizing in some way to make reading easier. 
- Figure 5, consider swapping position of 2nd and 3rd image for easier left to right reading- making the zoomed-out and zoomed-in images sit side by side.

### Questions
Q1: The pentagon prism structure appears to embed many facts about our base 10 counting system. Do you expect this tool-like structure to have been independently learnt by other models trained on base 10 math?

Q2: The pentagon prism has a 5 ring and 2 layers. The primes of 10 are 5 and 2. If the model had been trained in base 15, what shaped structure would you expect?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates how Transformers acquire long-range dependencies for multiplication via an implicit chain of thought (ICoT). Linear probing analyses indicate that ICoT encourages Transformer attention to cache pairwise products, providing evidence of long-range dependency capture. A 3D mapping of attention head outputs reveals nested representations of number tokens. The authors also propose a loss function that enables Transformers to learn multiplication efficiently without ICoT.

### Strengths
- The paper thoroughly examines how a Transformer with ICoT processes input digits to perform multiplication, supported by extensive experiments.
- Linear probing shows that the attention mechanism indeed stores intermediate computations.
- The geometric analysis reveals an interesting pentagram-like structure.

### Weaknesses
The paper has unclear points in several critical areas (premise, definitions, and mathematical exposition), which made it difficult to fully appreciate the contributions.

---
**Premise and baseline model.**

First, while the authors claim that SFT models cannot learn multiplication, prior work reports that simple GPT models readily learn multiplication when using zero-padding (to fix the number of digits) and digit reversal. Indeed, [1] shows that GPT-2 small successfully learns multiplication up to 15×15 digits. The present paper appears to use the same technique, which was not used in [Yang et al., 2023]—the work cited as evidence that multiplication is hard to learn. Consequently, the premise of this paper seems to rest on an unfair contrast: the claimed hardness relies on [Yang et al., 2023], whereas the technique employed follows [1], which already addresses the hardness.

[1] Shen et al., "Positional Description Matters for Transformers Arithmetic," 2023.

Second, it is unclear what “SFT models” refers to. No precise definition is given. Line 71 only states “a model trained with standard fine-tuning,” but if it is fine-tuned, on what pretraining data? What is the architecture—encoder–decoder or decoder-only? In either case, why does the baseline fail to learn multiplication unlike in [1]? Reversing target digits already encourages a certain chain of thought. The baseline should be able to reach near-perfect accuracy on the target task (i.e., 4×4-digit multiplication).

---

**Presentation**

The presentation obscures several core results.

The main text states that Figure 3 shows mean absolute error, but the scatter plots do not appear to reflect that. What are the diagonal points? In the upper row, why are blue points clustered at the bottom right?

Section 4 is also difficult to follow due to many undefined or cluttered symbols and operators.
- [Eq. 4] $A, B, \oplus$ are undefined. If $A$ is a matrix, the Minkowski sum between a matrix and a scalar $\epsilon$ is not defined. Assuming broadcasting, the right-hand side is a matrix while the left-hand side is a set, so the statement “matrix includes a set” is not meaningful.
- [Eq. 5] $\mathrm{Cov}(A_i)$ is undefined. Is this the covariance over entries in $A_i$, or across the index $i=1,2,\dots$? It is also unclear why $\Sigma_A$ and $\Sigma_B$ should share eigenvectors. The “local” covariance $\Sigma_{\mathrm{local}\mid a_i}$ appears to be conditioned on $a_i$, but its definition is not.
- [Fourier expansion] The index $k$ and the operator $*$ are undefined. $\mathbb{1}(n)$ and $\mathbb{p}(n)$ are not defined; the notation suggests vectors, but other terms are not vectors, which is inconsistent. What does $\mathbb{1}(n) \equiv 1$ mean?

---
**Insights**

While the paper aims to explain why multiplication is hard for SFT but feasible with ICoT, the results do not fully support this claim. Linear probing shows that how the successful model (i.e., the ICoT model) processes the digits while the SFT model does not. This finding is interesting but describes what happens in successful versus unsuccessful models rather than explaining what causes success or failure. Section 5 states:

> Despite only the middle digits receiving gradients (as they are the only sources of loss remaining), their losses plateau, suggesting that the model is stuck in a local optimum that lacks the long-range dependencies to properly learn the middle digits.

However, attributing failure to being stuck in a local optimum—without theoretical justification—does not provide a satisfactory explanation.

---

Overall, the work appears to rest on an incorrect (or unfair) premise that SFT models cannot learn even 4×4-digit multiplication, thereby overestimating the task’s hardness and overstating the benefits of ICoT. In addition, the presentation requires substantial improvements. Finally, the claim to explain “why” should be made with greater care to avoid overstatement.

### Questions
Please address the weaknesses raised above.

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper studies why certain models (standard fine-tuned) fail at successfully performing multi-digit multiplication. The authors gain insight into this by reverse engineering a model that *can* successfully perform multi-digit multiplication via implicit chain of thought (ICoT). They uncover that the successful ICoT model encodes long-range dependencies, while the standard fine-tuned (SFT) models seemingly do not. Given these insights, they develop an auxiliary loss that provides a helpful inductive bias for the SFT model to learn long range dependencies to perform multi-digit multiplication.

### Strengths
1. Develop an understanding that leads to an actionable intervention: the auxiliary loss that leads to the SFT model getting up to 99% after previously being unable to perform the multiplication task.
2. The paper is well-written and the visualizations are well-made.
3. The authors include code with their paper.

### Weaknesses
W1. The present study is only conducted on 4-digit multiplication. Additionally, it's unclear how the auxiliary loss would generalize to multiplication with more digits e.g. 5x5, 6x6, etc.

W2. It's unclear how the insights would generalize to other tasks.

### Questions
Q1. Related to weakness 1: Do you have general results on multiplying numbers with more than 4 digits?

Q2: Do you have any idea why the case with the auxiliary loss goes up to 99% but not to 100% like the ICoT model? How does the auxiliary loss model perform as the number of digits increases? Does it stay at 99% consistently as the length increases or are there any changes?

Q3: How sensitive/robust is the performance with the auxiliary loss as you vary hyperparameters and random seeds?

Q4: I'm not sure if this is known in the literature already, or if you have thought about this, but do you have any insight on why the SFT model fails at the task in the first place? My understanding is that you diagnose the failure being related to the difficulty of the middle digits, but is there any understanding as to why the transformer with an autoregressive loss isn't able to do this?

Note: I find it intriguing that you use methods that on their own are not always robust for mechanistic understanding of exact model behaviour (e.g. PCA, linear probes, logit attribution, attention patterns), etc. but it still gives you insight to something important the model is doing because you're able to take a model that was previously unable to solve the task and significantly boost its performance on the task. Perhaps I'm not familiar enough with the mech interp literature in this area, but I personally find this interesting and it raises some philosophical questions for me on what is ``important'' for mechanistic understanding.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies Transformers’ ability to learn tasks with long-range dependencies, focusing on 4x4 multiplication.
It compares a standard fine-tuned model (SFT) with a model trained using implicit chain-of-thought (ICoT).
With a 2-layer, 4-head architecture, ICoT reaches 100% accuracy, whereas SFT remains below 1%.
The core of the paper is a comparison of the internal mechanisms learned by SFT vs. ICoT. 
Using logit attribution and linear probing, the authors argue that ICoT captures the required long-range dependencies while SFT does not.
To explain how ICoT computes these dependencies, the paper introduces an attention tree, revealing a sparse, binary-tree-like routing pattern that supplies the necessary tokens to compute $c_2$.
The paper then studies the geometry of ICoT’s hidden representations via PCA. 
It finds that intermediate representations cluster by $a_i$ and $b_j$, and that the final hidden states exhibits a pentagonal-prism structure.
Additionally, the authors observe that SFT appears to get stuck in a local minimum, based on gradient norms and per-$c_k$ losses over training.
Finally, they introduce an auxiliary loss to predict $\hat{c}_k$, which enables SFT without CoT to learn the task successfully.

### Strengths
**S1.**
The paper is well written, with sufficient methodological detail.

**S2.**
Claims are validated through multiple analyses (logit attribution, linear probes, attention tree visualization, PCA, gradient norms and losses), which together provide strong support.

### Weaknesses
**W1.**
The paper does not fully address the fundamental reason why Transformers fail to learn multiplication when trained with SFT. 
The results convincingly indicate that SFT fails to capture long-range dependencies and that ICoT succeeds, but they do not explain why SFT fails to develop those dependencies in training (e.g., optimization landscape, inductive bias).
For example, in line 375, the paper states that “the model is stuck in a local optimum”, and does not clarify the underlying cause.
At the current stage, the work feels closer to “an analysis of differences between SFT and ICoT”, than to a full answer to the title “why can’t transformers learn multiplication”.

Aside from this limitation, the work is interesting and solid.

### Questions
**Q1.**
I’m slightly unsure how Figure 5 was produced. 
My understanding is: (1) run many problems; (2) collect the output vectors of a specific first-layer attention head; (3) run PCA; (4) color points by the digit at $a_i$ or $b_j$.
Which timestep is used for extracting output vectors?
Also, what does the purple box indicate in Figure 5(b)?

**Q2.**
How does Section 4 relate to the main message that ICoT learns multiplication while SFT fails? 

**Q3.**
The pentagonal prism in Figure 6 is interesting.
Does this structure consistently appear across different ICoT runs (e.g., different seeds/initializations)?
Also, if the task were posed in a different base (e.g., 11 (prime) or 30 (many divisors)), what geometry would PCA reveal? (the second question is just out of curiosity, so you don’t have to run extra experiments for this question.)

**Q4.**
In Section 6, the SFT model with the auxiliary loss solves 4x4 problems successfully.
Could you provide Figure 2/5/6 style visualizations for this model? 
This would help assess whether its internal mechanism aligns with ICoT or differs in important ways.

### Soundness
3

### Presentation
3

### Contribution
2
