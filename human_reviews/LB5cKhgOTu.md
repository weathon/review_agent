# QERA: an Analytical Framework for Quantization Error Reconstruction

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 6, 8, 8, 6, 6

## Abstract
The growing number of parameters and computational demands of large language models (LLMs) present significant challenges for their efficient deployment.
Recently, there is an increasing interest in quantizing weights to extremely low precision while offsetting the resulting error with low-rank, high-precision error reconstruction terms.
The combination of quantization and low-rank approximation is now popular in both adapter-based, parameter-efficient fine-tuning methods such as LoftQ and low-precision inference techniques including ZeroQuant-V2.
Usually, the low-rank terms are calculated via the singular value decomposition (SVD) of the weight quantization error,
minimizing the Frobenius and spectral norms of the weight approximation error.
Recent methods like LQ-LoRA and LQER introduced hand-crafted heuristics to minimize errors in layer outputs (activations) rather than weights, resulting improved quantization results.
However, these heuristic methods lack an analytical solution to guide the design of quantization error reconstruction terms.
In this paper, we revisit this problem and formulate an analytical framework, named Quantization Error Reconstruction Analysis (QERA),
and offer a closed-form solution to the problem.
We show QERA benefits both existing low-precision fine-tuning and inference methods --
QERA achieves a fine-tuned accuracy gain of $\Delta_{\text{acc}}$ = 6.05\% of 2-bit RoBERTa-base on GLUE compared to LoftQ;
and obtains $\Delta_{\text{acc}}$ = 2.97\% higher post-training quantization accuracy of 4-bit Llama-3.1-70B on average than ZeroQuant-V2 and $\Delta_{\text{ppl}}$ = $-$ 0.28 lower perplexity on WikiText2 than LQER.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Quantization Error Reconstruction Analysis (QERA), an analytical framework for reconstructing quantization error with low-rank terms. QERA offers a closed-form solution to the problem of quantization error reconstruction and demonstrates that LLMs quantized with QERA achieve better linguistic capabilities compared to previous approaches.

### Strengths
1.	It introduces an analytical framework for reconstructing quantization error by considering the layer output, not just the weight values.
2.	It provides detailed mathematical proofs.

### Weaknesses
1.	Although the paper claims that demonstrating the relationship between minimizing layer output error and minimizing model output error is a key contribution, this claim seems overstated. Many existing quantization approaches [1, 2] for LLMs already consider layer output error rather than focusing solely on weight approximation error to achieve more accurate quantization.
2.	The overhead of considering layer output is not sufficiently discussed and is only briefly mentioned in Figure 12(b) of the appendix. This paper lacks a thorough analysis of the overhead (e.g., memory requirements and runtime) associated with the error reconstruction procedures. While weight approximation error has limitations in reconstructing errors, it offers more efficient procedures since it does not require a calibration dataset. However, approaches that consider layer input/output for error reconstruction do require a calibration dataset and more computation. Despite this discrepancy in calibration overhead, the paper does not provide a comprehensive analysis of the error reconstruction overhead in the proposed method compared to previous works.
3.	According to Figure 12(b) of the appendix, as QERA-exact appears to be very slow, it seems fair to compare LQER and QERA-approx. However, QERA-approx offers minimal advantage over LQER in terms of quantization.
4.	The paper does not compare the proposed method with LQ-LoRA, a more advanced method that also uses a calibration dataset for error reconstruction. A comparison between the proposed method and LQ-LoRA is important for evaluating QERA.
5.	The paper uses confusing terminology when categorizing experiments. It should use standard terms such as "quantization-aware fine-tuning" and "post-training quantization" to classify experiments, rather than "fine-tuning experiments" and "quantization experiments," as fine-tuning experiments also include quantization.

[1] Frantar, Elias, et al. "Gptq: Accurate post-training quantization for generative pre-trained transformers." ICLR 2023.

[2] Lin, Ji, et al. "AWQ: Activation-aware Weight Quantization for On-Device LLM Compression and Acceleration." MLSys 2024

### Questions
Please check the Weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper directly gives the analytic solutions of the linear compensation matrix for quantization error correction.  The analytic solutions involve both exact and approximate solutions to tradeoff compensation quality and efficiency. The analytic solutions are solidly validated for two scenarios, post-training and fine-tuning-based quantization.

### Strengths
1. The work formally demonstrates that optimizing to reduce the model layer output error is more effective than minimizing the weight quantization error. Given the fundamental importance of the optimization objective in this field, the work has a significant impact.

2. The assumption used to approximate the exact solution appears reasonable and can be validated through practical testing.

3. The experiments conducted in the work are extensive and comprehensive.

### Weaknesses
1. The efficiency of the exact and approximate solutions, like computational complexity and the concrete execution time compared with related works, should be further clarified.
2.  Eq. (18), $\mathop{\arg\min}_{Q_k} $. 
3. Can the analytic solution be used in LLM pruning? Given that pruning and quantization usually have similar mathematical modeling to some extent.

### Questions
Please see the weaknesses.

### Soundness
4

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
4

### Summary
In this paper, authors give the  analytical solution to solve the minimizing the errors in layer outputs via low rank terms. And authors prove that  minimizing the layer output error is better than minimizing the weight approximation error from the aspect of model performance.

### Strengths
This article is excellent in aspect of motivation, problem solving, and paper writing, and is also highly recommended for  its algorithm engineering work. 
1. In terms of motivation, this article chooses to use theoretical methods to solve problems that can only be solved using heuristic algorithms at this stage, and determines the theoretical extreme value of the problem and the method to reach it. 

2. This article provides a very solid analytical method and gives an algorithm for solving the extreme value according to this method, which is solid and reliable. 

3. Paper writing aspect, this article has been detailed and concise, simplifying the repetitive proof of Theorem 2 and placing the proof of Theorem 1 as an important part of the text, allowing readers to fully understand the contribution of this article. In addition, the structure of this article is clear, and the questions and answers are clearly stated. 

4.The biggest advantage of this article is that it reasonably engineers the algorithm and provides a reasonable simplified algorithm that is easy to implement in engineering.

### Weaknesses
The authors insight that minimizing the output error is better than weight approximation error is is consistent with our practical experience in the aspect of model performance. However, this point is hard to prove via experiments, because we cannot enumerate all weight approximation methods on every models. The conclusion is so strong. 
So, two suggestions are that 1. give a mathematical proof of this point. 2. avoiding discuss this conclusion in paper, and only show your work is better than SOTA low rank methods.

### Questions
1.How to prove that the reason of worse performance of weight approximation methods is rooted in the model's nature instead of your weight approximation methods choice in experiments?

2.In first paragraph, k<<min(m,n). I think it should be k << $\frac{mn}{m+n}$ because the computation cost of W is mn, the computation cost of $A_KB_K$is mk+nk, we want (m+n)k<mn, k should satisfy k <$\frac{mn}{m+n}$

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes QERA that analytically finds the proper low rank terms by minimizing the layer output error. They show that minimizing the layer output error is more closely related to minimizing the model output error than minimizing the weight approximation error. They empirically demonstrate their method is better than other previous methods in both QPEFT and PTQ perspectives.

### Strengths
1. The paper is generally well-written and easy to follow.
2. The idea of deriving the analytical solution to the low rank terms by minimizing the layer output error is new.

### Weaknesses
The weaknesses of this paper mostly come from the experiment part.

1. The numbers in Table 1 and Table 2 don't match with the loftq original paper. Is that because you change the experimental setup? Could you please show your method outperforms loftq in their setup?
2. In the original loftq paper, they includes some experimental results about 2bit fine-tuning. Could you also show some results about 2bit fine-tuning?

### Questions
N/A

### Soundness
2

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
4

### Summary
This work proposes a framework for analyzing the quantization error after it is compensated using low-rank, high-precision terms. In such a decomposition, a weight matrix $\mathbf{W}$ is approximately split as $\mathbf{\tilde{W}} + \mathbf{A}_k\mathbf{B}_k$, where $\mathbf{\tilde{W}}$ is the quantized weight, and $\mathbf{A}_k\mathbf{B}_k$ is the low-rank compensation. This work aims to minimize the Frobenius norm error of the outputs of each layer (in contrast to just the weight quantization error), and propose closed-form solutions for the low-rank terms. The improved benefits are validated with numerical experiments of both encoder-only (RoBERTa) and decoder-only (LLaMa family) models. The experiments involve both post-training quantization (PTQ), as well as parameter-efficient fine-tuning (PEFT).

### Strengths
The paper analytically considers the problem of compensating the quantization error using low-rank high-precision components. The paper is generally well-written, although the work will benefit if it takes into account and compares with more recent works which takes into account the same problem (see weaknesses below).  

The numerical experiments are comprehensive, and the results on a wide variety of models are presented. They are also compared with some other prior works, and show improved benefits.

### Weaknesses
My major concern with this paper is that it fails to take into account more recent works in this area, and justify how it compares with those works. The contribution of not really clear in light of a more recently proposed algorithm, Caldera (https://arxiv.org/abs/2405.18886) solves the optimization problem (9) optimally, i.e., the output error is minimized and closed form solutions for the low-rank factors are obtained (ref. Lemma 4.2 in the paper). Could the authors highlight the difference in their result of QERA-exact solution (Thm. 1) with Lemma 4.2 (Caldera)? 

Furthermore, the autocorrelation matrix, $\mathbf{R}_{\mathbb{XX}}$ (which is also referred to as Hessians, because it is the Hessian of the quadratic loss in (9)), does need to be computed using a calibration dataset -- but this is a one-time cost. Additionally, the approximation in Assumption 1 for QERA-Approx approximates the autocorrelation matrix as a diagonal matrix -- this is not necessarily true as shown in Figs. 5, 7 and 8. 

Secondly, minimizing error in layer outputs for PTQ is not really a recent idea as mentioned in the paper. It has been around for a few years now. See for example, https://arxiv.org/abs/2004.10568. 

Despite the weaknesses mentioned here, the numerical evaluations on the models and the regimes of 3/4-bit quantization are most likely new.

### Questions
Please see the Weaknesses section. I would be happy to readjust my score if they are satisfactorily addressed.

### Soundness
3

### Presentation
3

### Contribution
2
