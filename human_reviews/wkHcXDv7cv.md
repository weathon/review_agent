# Tuning Frequency Bias of State Space Models

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6, 8

## Abstract
State space models (SSMs) leverage linear, time-invariant (LTI) systems to effectively learn sequences with long-range dependencies. By analyzing the transfer functions of LTI systems, we find that SSMs exhibit an implicit bias toward capturing low-frequency components more effectively than high-frequency ones. This behavior aligns with the broader notion of frequency bias in deep learning model training. We show that the initialization of an SSM assigns it an innate frequency bias and that training the model in a conventional way does not alter this bias. Based on our theory, we propose two mechanisms to tune frequency bias: either by scaling the initialization to tune the inborn frequency bias; or by applying a Sobolev-norm-based filter to adjust the sensitivity of the gradients to high-frequency inputs, which allows us to change the frequency bias via training. Using an image-denoising task, we empirically show that we can strengthen, weaken, or even reverse the frequency bias using both mechanisms. By tuning the frequency bias, we can also improve SSMs' performance on learning long-range sequences, averaging an $88.26\\%$ accuracy on the Long-Range Arena (LRA) benchmark tasks.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper addresses the frequency bias of State Space Models (SSMs) and provides methods to adjust this bias. The authors introduce a formal framework to characterize the frequency behavior of SSMs and reveal that standard SSMs, as commonly described in recent literature, tend to learn low frequencies. While this low-frequency bias is often advantageous, the authors argue that it is possible to improve the model's frequency response.

To achieve this, they propose two different approaches: the first modifies initialization to control which frequencies the model can learn and the second alters the transfer function to adjust how much the model emphasizes particular frequencies.

Although I am not a expert in SSMs, I found the paper well-structured and effective in explaining complex issues with clear, illustrative examples.

### Strengths
Well-written with effective illustrative explanations.

Provides a theoretically sound analysis of a complex problem.

Practical relevance is high.

### Weaknesses
The analysis is limited to diagonal systems.

Tuning the additional hyperparameters, α and β, may pose challenges in practice.

### Questions
Could you provide some guidelines for tuning the α and β parameters? How much tuning was necessary to achieve the results in Table 3?

How could the analysis be extended to non-diagonal systems?

### Soundness
4

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
4

### Summary
This paper studies the frequency bias of state space models, where the authors identify a tendency, under generic initialisations of the state space matrix, to favor the response to low frequency signals. The authors give a definition of the frequency bias through the total variation of the transfer function, in that a low TV over an interval in frequency space is understood as a bias to not capture this region effectively. The authors then show that generic initialisations indeed gives lower TV for high frequency regions, and moreover that this cannot be fixed by optimisation (in the sense that the gradient on the frequency component is small if there is little initial overlap). 

Two solutions are proposed to mitigate this issue: better initialisation where a hyper-parameter is introduced to weigh higher frequencies differently, or changing the loss function by adding a weighting function in frequency space to promote training larger frequencies (or vice versa). Improvements are observed on both a synthetic denoising task and benchmarks from long range arena.

### Strengths
The paper is very clearly written, and the motivation and implications of the theoretical results are clearly explained. The identification of the frequency bias is useful for problems where high frequency components need to be extracted. The proposed methods to mitigate the frequency bias appear simple to implement, and are principled.

### Weaknesses
1. The paper can be improved by clarifying some notational and definition issues, and these are detailed in the questions below.
2. Notational suggestion: the notation of using $y_j$ to denote imaginary part of the diagonal of $A$ can be changed to avoid confusion with the output of the network, also called $y$.
3. Since the SSM transfer function is differentiable, it may be simpler to define the total variation as the integral of its derivatives to avoid unncessary complications.

### Questions
1. Can the authors explain more clearly why the rate of change of the transfer function is a proper measure of frequency bias? Traditionally (e.g. low pass filters) would consider the support of the transfer function, or its magnitude over frequency intervals, as a measure of its response to frequency components. Some proper definition of "frequency sensitivity" would make the definition of frequency bias in terms of TV more transparent.
2. After proposition 1, Rule II: what happens in the limit $L\to\infty$?
3. Eq 7: I find it strange that the weighting factor is $(1+|s|)^{2\beta}$ instead of $(1+|s|^2)^{\beta}$, as in the usual Bessel potential spaces. What's the motivation for this departure? In any case, the name "the Sobolev norm" is unclear since there are many Sobolev spaces. 
4. For the mitigation algorithms, what are some general tuning strategies for $\alpha$ and $\beta$? Fig 5 seems to say $\alpha$ is more sensitive than $\beta$. Some comments on this would be helpful.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this paper, the authors find that when training state space models with canonical initialization methods, there is an implicit bias such that the models are implicitly toward to capture low-frequency components instead of high-frequency ones.
To encourage the models to capture a more broader frequency pattern, the authors propose two mechanisms: one is scaling the initialization and another method is to apply a Sobolev-norm-based filter on the state space models.
By tuning the frequency bias, the state space models are shown to be able to have better performance on the long range arena benchmark.

### Strengths
1. The organization and presentation of this paper is smooth and clear, and it provides a better understanding on the training mechanisms of state space models on sequence modeling.

2. I find the paper to be well-written and easy to follow. The overall topic of state space models is important.

### Weaknesses
1. In the end of Section 2, the authors state that the larger the total variation of the transfer function is, the better an LTI system is at distinguishing the Fourier modes. This claim is not intuitive for me, and the total variation cannot distinguish the following two different cases: the first one is large amplitude with low frequency and the second one is small amplitude with high frequency. For example, $G$ is an impulse response vs $G$ is a sinusoidal wave with a small amplitude. From my understanding, these two LTI systems have different ability on distinguishing the Fourier modes.

2. The statement after Lemma 1 that small $|y_i|$ induces small total variation seems to be wrong. From the upper bounds in Lemma 1, if $|y_i|$ decreases, the the upper bounds increase, which means that the total variation will be large.

3. The initialization method in Corollary1 is not a commonly used method. For S4D, the initialization methods for $a$ are mainly S4D-Legs and S4D-Lin. Why not choose these two initialization methods instead?

### Questions
For the numerical experiments (Table 3) on the long range arena benchmark, how is the $A$ matrix initialized? Is $A$ initialized by the method in Corollary 1? There are some ablation studies on the scaling of the imaginary part of $A$ in the S4D paper [1]. It is shown in [1] that scaling all imaginary parts by a constant factor substantially reduce the performance of S4D, so I am curious that if we only scaling the imaginary part of $a$, does it help to improve the performance on the long range arena benchmark? It would be more convincing if the authors can provide more experiment results of only scale $a$, only train $\beta$, the results in Table 3 is the case for scale $a$ + train $\beta$.






[1] Gu, Albert, et al. "On the parameterization and initialization of diagonal state space models." Advances in Neural Information Processing Systems 35 (2022): 35971-35983.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies the frequency bias for State-space-models (SSMs). With strategies to select better initialization and Sobolev filtering (during training), this problem could be mitigated. In the experimental study, this manuscript also provides extensive results to show the effectiveness of their proposed strategies.

### Strengths
This paper presents a remarkable analysis of the frequency bias in state-space models (SSMs). It has been commonly observed that neural networks tend to fit low-frequency information first, often filtering out high-frequency data in many cases. This paper critically emphasizes the importance of initialization and introduces a strategy to mitigate this problem using a Sobolev filter. The figures presented in the experiments are highly inspiring. As a theoretical work, this paper provides sufficient experimental results, demonstrating strong performance.

### Weaknesses
The weaknesses of this paper mostly come from two parts. 

1. The analysis is based on SSM which is potentially hard to be generalized on general neural networks. 

2. Even though the reason is getting clearer, these strategies of this paper could still be suboptimal for practitioners.

### Questions
1. The authors may need to define HiPPO in line 92 Page 2 before mentioning it. 

2. How will these discoveries help for general neural network settings? 

3. Would the Sobolev filter difficult to compute in practice (when doing the SSM training)? 

4. How to interpret Figure 4 when having low-freq. noise and high-freq. noise. What is the difference? Additionally, why low-freq. noise can be denoised (with higher-frequency-pass filter) but the images in the bottom rows will be reconstructed?

### Soundness
4

### Presentation
4

### Contribution
4
