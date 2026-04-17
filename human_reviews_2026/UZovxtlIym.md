# Data Augmentations for Arithmetic Length Generalization in Small Transformers

- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Transformers often struggle with achieving length generalization on algorithmic tasks. To date, the most successful techniques attempting to achieve length generalization impose modifications to the model architecture. Instead, we propose Aligned Blankspace Augmentation (ABA), a simple data augmentation method that zero-pads numbers and inserts synchronized blank spaces across operands, demonstrating that the original Transformer architecture can achieve length generalization. Experiments demonstrate that small Transformers trained on up to 20-digit addition using our method achieve high accuracy on 200-digit problems, significantly outperforming prior works. The approach also enhances performance on other tasks like sorting and multi-operand addition, and improves multiplication generalization with scratchpads.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a novel and compelling approach to the challenging problem of length generalization in Transformers for algorithmic tasks. The authors introduce ABA, a data augmentation technique that uses synchronized zero-padding and blank space insertion to enforce alignment between corresponding digits across operands. The central and powerful claim is that robust length generalization can be achieved without modifying the Transformer architecture. Also, the paper is supported by extensive experiments and a non-trivial theoretical analysis.

### Strengths
1. The proposed ABA method is remarkably simple yet effective, demonstrating that a purely data-centric intervention can outperform sophisticated architectural modifications like Position Coupling and Abacus Embeddings.
2. The experimental evaluation is thorough. The paper validates ABA's effectiveness not only on the canonical task of multi-digit addition, demonstrating generalization from 20 to 200 digits, but also on a diverse suite of algorithmic tasks including copying, reversing, sorting, and multi-operand addition.
3. The authors provide a principled explanation for why ABA works while random spacing does not. However, I have some questions about the proof.

### Weaknesses
See questions.

### Questions
1. ABA constructs training data by inserting spaces to extend the original numbers to a length of $p_{max}$. However, the paper mentions that the generalized length cannot exceed this constructed $p_{max}$. Since we're constructing data anyway, how would directly constructing data with longer operands perform? Would it be better than ABA? If so, could the authors explain the unique advantages of ABA? (Or are there other tasks where directly constructing longer data for training isn't feasible?)
2. In Table 2, ABA doesn't show a significant advantage compared to Position Coupling across the five tasks. Could this be related to the characteristics of the tasks, meaning some tasks are more suitable for enhancement using the ABA approach? Furthermore, the performance difference between the 'fixed' and 'var' variants is quite pronounced. What is the reason for this disparity? Is it due to inherent variability?
3. I didn't fully understand the application of ABA to the sorting task. Where should the newly added "operand placeholders" be placed in the answer? Also, in Figure 5, why does the raw data for the sorting task have three numbers (a, b, c) on the left side of the equals sign, but four on the right?
4. Regarding "Prop. 6.1", I believe the communication complexity should be at most $O(\log^2 p_{max})$. To communicate the padding map, one only needs to send the positions of K digits. Transmitting the position of a single digit requires $\log(p_{max})$ bits, so the total should be $O(K \log p_{max}) = O(\log^2 p_{max})$ bits, assuming $K = O(\log p_{max})$. The proof in the paper for "Proposition H.2" cites a conclusion from Roughgarden, 2015. That work doesn't have the $p_{max} >= 2^K$ restriction, so its worst-case scenario doesn't seem directly representative of the situation here.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper studies how to achieve length generalization in Transformers on arithmetic tasks.
While Transformers perform well on lengths seen during training, they often struggle on unseen lengths.
Prior work mainly improves length generalization through architectural changes or specialized positional encodings. 
This paper proposes Aligned Blankspace Augmentation (ABA), a data augmentation strategy that inserts blank spaces at identical relative indices within operands and results.
With zero-padding and synchronized space insertion, the paper introduces two variants: ABA-fixed and ABA-var.
Using ABA-fixed with RPE, Transformers trained on addition with operand lengths up to 10 digits successfully generalize to up to 120 digits.
With ABA-var or with other positional encodings, the models still generalize, but less strongly overall.
The paper also applies ABA to other algorithmic tasks (Copying, Reversing, Multiplication, Multi-operand addition, Sorting). 
With appropriate formatting (including placeholder operands), the trained models can generalize with respect to the number of operands.
Finally, the paper demonstrates length generalization on multiplication using a carefully designed scratchpad.

### Strengths
**S1.**
The paper is clearly written and provides thorough experimental details and comparisons to prior work.

**S2.**
The proposed method achieves strong length generalization across multiple tasks, especially in addition with ABA-fixed + RPE.

### Weaknesses
**W1.**
My major concern is that the proposed method requires exposing the model to long sequences during training.
Although the raw data are short, after augmentation the effective sequence length is set to a target $p$ that matches the test-time maximum via $p_{max}$.
This seems undesirable for a method intended to enhance length generalization.
If one has sufficient capacity to train on long sequences, why not simply train the model directly on longer sequences?


**W2.**
In Table 2, I was first surprised that ABA-fixed achieves length generalization for multi-operand addition task without a scratchpad, since the model has never added more than five single-digit numbers at once during training.
Even with placeholder operands, the model has not encountered sums with >5 single-digit addends during training, and thus I feel the result highly nontrivial.
For example, consider the problem 19+19+19+19+19+19+19=133.
To produce the correct result, the model should propagate a carry of 6 from the least-significant digit (9x7=63), whereas during training the largest carry from that column would be at most 4 (e.g., 9x5=45).
Therefore, I suspect that the proposed method may not faithfully length generalize for multi-operand addition task, especially where digits are concentrated in $\{8,9\}$ and carries become large.
Could the authors report results on such high-carry, high-operand-count problems (e.g., 6-10 operands with digits biased to {7,8,9})?
Testing longer per-operand digit lengths would be ideal, but for this stress test, keeping operand lengths within the training distribution is fine, as my concern here is whether the model can truly achieve generalization along the number of operands axis.

### Questions
**Q1.**
For the basic addition task, could the authors report accuracy specifically on long carry-chain cases (e.g., 999999999999999+1)?
This rare pattern can test whether the model truly learns carrying across the entire length.

**Q2.**
I don’t understand the sorting example in Figure 5.
Why does a $d$ appear in the output before augmentation (between Line 275 and 276)?

**Q3.**
In “Generalization from 10+10”, the paper claims ABA-fixed learns addition when trained only on 10-digit problems, whereas Position Coupling and Abacus fail.
In my personal view, the superior performance of ABA-fixed “primarily” stems from its methodological property that the training sequence length matches the test sequence length.
This is further reflected in ABA-var (Figure 9): the model generalizes to 16-digit inputs, but fails below 10 digits, since samples with $p < L (=10)$ never appear during training.
Therefore, I think this experiment does not strongly support the claimed superiority of the proposed method; the result appears highly trivial because the task is inherently advantageous for ABA-fixed.
I would like to hear the authors’ thoughts on this argument.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper argues that a data augmentation technique called ‘Aligned Blankspace Augmentation (ABA)’ can improve the length generalization capability of Transformers in solving an arithmetic task. The ABA technique is summarized as a method of inserting ’blank spaces (a single non-digit non-operator token)’ between some digits of the operands and the answer, matching their relative positions. The blank space is used to match the trained sequence length to be (up to) the test length. For the integer addition task, the method is applied alongside a zero-padded, reversed format. The main method is compared with other state-of-the-art baselines, such as Position Coupling and Abacus Embedding, for length generalization in arithmetic tasks. As a result, the paper reports ‘ABA-fixed+RPE’ as their best method. The ABA strategy is also evaluated on other tasks such as multiplication (with/without scratchpads), copy/reverse, and more. Lastly, the paper also provides a communication-complexity-theory-based result explaining why the aligned blank spaces are beneficial in solving K-length addition with a limit transformer.

### Strengths
1. The paper is well-written. It clearly describes its main methodology, with great detail about the training and evaluation methods.
2. The paper provides abundant and quality experimental results to support their main claim, comparing with state-of-the-art training methods and data formats in the literature of transformer’s length generalization for arithmetic tasks.
3. The paper also provides a theoretical result to supplement their argument about the benefits of aligned blank spaces. (Honestly, I didn’t fully validate the proof yet. However, the statements are intuitively correct, at least in a high-level sense. Also, they provide meaningful insights on why the aligned blank spaces possibly enhance the trainability of a model towards a length-generalizable one. Hence, I put this point as a strength. I will investigate the proof after submitting this review.)

### Weaknesses
1. Speaking of ABA-fixed+RPE, which is the only baseline that beats all the existing methods (including Position Coupling and Abacus), I am afraid I cannot agree that this method is achieving the **length generalization**. Of course, I agree that the proposed training method is achieving a form of out-of-distribution generalization because the test data distribution is very different from the train data distribution (i.e., all the test instances lie outside of the support of the trained distribution). Nevertheless, these two distributions are identical in sequence length. In fact, I expect that ABA-fixed is not a length-generalizable method at all. Thus, I can only agree that ABA-var is achieving some form of length generalization, although it does not show a strictly better performance compared to the other baselines. Nonetheless, I think the authors can motivate their idea by taking ABA-fixed as an ‘oracle’ method (ideal-but-nonsense), instead of arguing it as their main methodology for length generalization.
2. The methods are not compared properly in terms of training cost. First of all, as admitted by the authors, the ABA strategy takes a much larger training cost than the other baselines because it requires training the model on (up to) target sequence length $p_{\rm max}$. Since the target length is often taken as ~10x larger than the training length (e.g., train up to 20, test up to 200), the computation cost for attention calculation must be up to ~100x larger in theory. Despite such a training inefficiency, a fair comparison between methodologies can be done if the training cost is matched strictly and manually (e.g., using a different number of steps or different mini-batch sizes to match the number of tokens seen during training or the training time). However, training of every baseline is run for 50K iterations (except for Abacus, which is trained for 20K iterations). This clearly indicates that the comparison in the paper violates the fairness in terms of training cost. Indeed, Table 8 showcases that ABA takes ~4.7—5.8x longer training time and ~7.9—10.7x larger training memory than the other baseline methods. With access to this large training budget, I believe that direct training on the target sequence lengths is possible and will achieve a much better performance.
3. The proposed method seems unscalable. This is because, as the authors discovered, an excessively large hyperparameter $p_{\max}$ hurts the training stability and hence causes failure in length generalization. Nevertheless, the paper provides a justifiable reason for this phenomenon, and I believe a similar issue will happen in other baselines. I would say it is expected that a stable training with a larger $p_{\max}$ requires longer training lengths. Hence, I would rather suggest finding a synergetic combination between the training sequence length and the hyperparameter $p_{\max}$ that can achieve a much larger generalizable length. As an example, Cho et al. (2024, Appendix B.1) have achieved a significant length generalization up to 500-digit additions by training a (small) model up to 160-digit additions.
4. Lastly, I am afraid I cannot agree with the authors’ argument that all the existing techniques require architectural modifications. In particular, the works by McLeish et al. (2024) and Cho et al. (2024) do not necessarily require a modification in the model code, given that the implementation already supports an input parameter for custom position IDs: see the explanation in Appendix C of Cho et al. (2024). The claim is partially true, for example, when we require multiple levels of position IDs (Cho et al., 2025). However, since the main subject of comparison is a simple integer addition, I do not think that ABA is a better methodology in terms of architectural modification: all three methods, Abacus, Position Coupling, and ABA, require only a modification in dataset construction.

### Questions
1. One thing that I was quite surprised about the proposed method is that it allows inserting blank space in the middle of the operands and the answer. One problem that I expected was that it might hinder computing and passing the carries. I am a bit suspicious that the success of ABA is because the carry chain is not very long in most of the examples generated uniformly randomly. In other words, I am curious whether the method is also successful for the examples with long carry chains or not.
    1. I would like to suggest a synthetic task that simulates the necessity of an extremely long carry chain: a parity task with a scratchpad. Given a binary sequence $(x_1, …, x_L) \in \Sigma^L$ as a prompt ($x_i \in \Sigma:=\\{0,1\\}$ for each $i$), the answer is a sequence $(y_1, …, y_L) \in \Sigma^L$ of the same length, defined as $y_i = \sum_{j=1}^{i} x_i \bmod 2 = x_i +y_{i-1} \bmod 2$ (letting $y_0 = 0$). Will the ABA method be successful in achieving length generalization for this task? How about a general $m$-parity task (letting $\Sigma = \\{0, 1, …, m-1\\}$ and taking $\bmod m$ instead of $\bmod 2$)?
    2. Believe it or not, exactly because of this reason, I have *personally* tried before an idea of putting the same number of zeros on the left/right of the operands and the answer to match the target length, although I am currently not planning to publish this idea as a conference paper. It is quite similar to the setup of ABA-fixed, but there are two differences: I used zeros instead of blank spaces, and I did not insert the additional tokens in the middle of the numbers. I do not exactly remember the result, but it was quite good, even though I still do not believe that this success can be interpreted as length generalization either. Have you tried not to put the blank spaces in between digits? What is good for the originally proposed method above this option? Can you provide some numerical results comparing them?
2. I personally love the idea of fine-tuning the model with ABA-var+RPE, starting from the baseline model (without any tricks like Position Coupling or Abacus), because this seems to be a remedy for the large training cost requirement I pointed out above. How long does it scale (in terms of the generalizable length)? What is the minimal amount of fine-tuning to achieve a similar performance as a model fully trained with ABA-var+RPE?

### Soundness
1

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
The paper studies the problem of length generalization for arithmetics (addition and multiplication) with transformers. Several works have proposed solutions for this problem via changing the architecture (e.g., positional embeddings) or padding all of the inputs. This work claims that they can preserve the accuracy once they surpass the training lengths by none of the previous changes. Instead, they propose a novel augmentation method.

### Strengths
1- The paper is well-written and easy to understand. They have also included extra explanations and comparisons in the appendix that makes the paper complete. 

2- As far as I am aware, adding zeros for augmentation is investigated in previous work, but randomly adding blank tokens is novel. 

3- The experiments clearly show the benefits of using the paper's proposed method. They also cover most of the existing baselines in their comparisons that have made it more convincing. 

4- The results for ABA + scratch-padding in multiplication look promising considering that there are no architectural changes. 

5- The paper adapts the results of [Huang et al.] to explain why their method can address the length generalization problem in theory.

### Weaknesses
1- I am not certain about the distinction of augmentation with zeros that has been investigated before (e.g, multiplying both operands by 10^n) and the proposed solution that randomly adds blank spaces between digits (as opposed to only adding them to the two sides of the operands). I also did not find any arguments about this in the paper and I think this needs to be added. Even though the theory section justifies their results with the result of RASP, I think authors need to motivate this more intuitionally.  

2- The paper motivates their results by claiming for no architectural changes. However, as Table 1 shows, the performance indeed depends on the type of the positional encoding and this seems to be an important factor. 

3- The comparisons in the multiplication section is quite limited in the main body, and something similar to Figure 13 has to be added there.

### Questions
Pls see above for questions.

### Soundness
3

### Presentation
4

### Contribution
2
