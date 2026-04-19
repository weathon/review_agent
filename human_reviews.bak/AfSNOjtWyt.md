# Universal length generalization with Turing Programs

- Decision: Reject
- Scores: 3, 6, 5

## Abstract
Length generalization refers to the ability to extrapolate from short training sequences to long test sequences and is a challenge for current large language models. While prior work has proposed some architecture or data format changes to achieve length generalization, these proposals typically apply to a limited set of tasks. Building on prior scratchpad and Chain-of-Thought (CoT) techniques, we propose \emph{Turing Programs}, a novel CoT strategy that decomposes an algorithmic task into steps mimicking the computation of a Turing Machine. This framework is both universal, as it can accommodate any algorithmic task, and simple, requiring only copying text from the context with small modifications. We show that by using Turing Programs, we obtain robust length generalization on a range of algorithmic tasks: addition, multiplication and in-context SGD. We then demonstrate that transformers achieve length generalization on random Turing Programs, suggesting that length generalization is possible for any algorithmic task. Finally, we theoretically prove that transformers can implement Turing Programs, constructing a simple RASP (Weiss et al.) program that simulates an arbitrary Turing machine.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper proposes a method that combines effective positional encoding (Hard-Alibi) with a scratchpad technique to enhance length generalization in Transformers. The authors identify length generalization capabilities of copying task as a potential underlying factor contributing to this enhancement. Empirically, the paper explores the length generalization properties of the proposed methods through tasks involving addition ($n+n$), multiplication ($n\times 1$, $n\times 3$), $n$-sample SGD, and random Turing machines. Theoretically, it investigates the expressive power of Transformers simulating Turing machines.

### Strengths
This work presents a novel approach for improving length generalization by thoughtfully combining two existing methods. The experimental results on random Turing machines effectively demonstrate the proposed methods' applicability to more general tasks.

### Weaknesses
1. Although the paper effectively combines appropriate positional encodings with the scratchpad technique to enhance length generalization, it lacks comparative results against a broader array of positional encoding mechanisms and additional baselines under the same experimental conditions.

2. The theoretical section appears disconnected from the experimental findings. While the experiments concentrate on length generalization, the theoretical analysis solely illustrates the expressive power of Transformers in simulating Turing machines. The assumptions made are somewhat strong, and the result of $\exp(n)$ steps renders the theorem less impactful.

### Questions
1. Can you provide more experiments with additional baselines to compare length generalization performance under the same experimental setup, including:
   - Various positional encodings with and without the scratchpad.
   - Other prior approaches (e.g., those listed in Appendix B) for addition tasks.
   - Could the methods in Appendix B be applied to multiplication tasks? Would they exhibit non-trivial length generalization abilities?

2. Is it possible that combining effective positional encoding, the scratchpad technique, and improved data formats could yield even better length generalization for addition or multiplication tasks?

3. In line 250, what are the reasons for retaining only the most recent 500 tokens during testing? Is this crucial for achieving better length generalization results, considering that the training and testing context lengths are identical in this setup? An extensive ablation study appears essential to assess the impact of retaining versus dropping past tokens.

### Soundness
2

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
2

### Summary
A well-presented work on what is in its present state a toy deep learning problem.

### Strengths
1. Good writing, can be followed with relative ease.
2. Meticulous contextualization with respect to the previous work.

### Weaknesses
1. While a considerable part of Sections 1 and 2 has been dedicated to justification of the effort and the setting in the context of previous work, one cannot escape the feeling that the overall contribution and subsequent impact of this work are most likely to be minute. The setting of this effort is one where function-calling models already excel with unprecedented precision and at a considerably lower cost. To have a transformer model carry out arithmetic digit-by-digit, albeit in a learnable fashion, is a waste of computational resources, and hence this is not really a good set of experiments to consider.  While the work in its present state hints at a promise, it falls short of demonstrating any potential utility. A suggestion for the future direction of this effort would therefore be to attempt to address some other still "basic" but nevertheless useful tasks such as regular text manipulation or automated formal reasoning.
2. Minor - presentation focus:  The main text seems to assume familiarity with recent developments on algorithmic length generalization while dedicating an entire half-page to the well-known concept of Turing machines. I would suggest re-balancing.

### Questions
None.

### Soundness
4

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
Authors showed that a (very long) CoT which slightly change an input $x_i$ into $x_{i+1}$ up to a point where it solves a problem completely can achieve state-of-the-art results on arithmetic tasks. Authors showed that a bunch of problems can be solved with this approach. Finally authors showed that transformer can indeed implement Turing programs.

### Strengths
S1. It seems this is the first work which achieves a non-trivial length generalization on arithmetic problems without using data curation or representation tricks and customized positional embeddings to solve arithmetic task. 

S2. Result from Section 5.3 is pretty interesting because of the generalization capability it may bring.

S3. Empirical results on non-trivial tasks shows the effectiveness of their method.

### Weaknesses
W1. The strongest weakness is Theorem 4.1. The code is shown in Appendix D, but the construction proof is not justified at all. 

W2. Authors didn't spend any time on explaining why HAlibi embedding is strongly needed and what practitioners can do to solve similar math problems without using this embedding. In particular it's not obvious how to escape from embedding choice and rely only on the multiple CoT idea. 

W3. In general, ideas and theorems are not well motivated and explained. For example, only Figure 2 explains how to create a synthetic dataset for addition or Figure 4 for multiplication. 

W4. It's not obvious how to extend this idea to fairly similar arithmetic tasks, for example 20-multiplication. 

W5. (minor nit) some citations are reported twice, firstly in black and right after then in blue.

### Questions
Q1. Did authors think about other algorithm classes which can be learnt with a copying based CoT? 

Q2. Is there any limitation of this idea for longer sequence, i.e. going with a 3x generalization multiplier? Did authors collected some results there?

Q3. Are there additional tasks the authors think of to show the full potential of this idea? Maybe some graph related tasks?

### Soundness
3

### Presentation
2

### Contribution
2
