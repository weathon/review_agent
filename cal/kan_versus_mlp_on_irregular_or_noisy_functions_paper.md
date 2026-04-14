

{0}------------------------------------------------

# KAN VERSUS MLP ON IRREGULAR OR NOISY FUNCTIONS

Anonymous authors

Paper under double-blind review

## ABSTRACT

In this paper, we compare the performance of Kolmogorov-Arnold Networks (KAN) and Multi-Layer Perceptron (MLP) networks on irregular or noisy functions. We control the number of parameters and the size of the training samples to ensure a fair comparison. For clarity, we categorize the functions into six types: regular functions, continuous functions with local non-differentiable points, functions with jump discontinuities, functions with singularities, functions with coherent oscillations, and noisy functions. These features are typically not available as prior knowledge in real applications; therefore, we do not specifically select the corresponding network structure for each function. Our experimental results indicate that KAN does not always perform best. Furthermore, increasing the size of training samples can improve performance to some extent. When noise is added to functions, the irregular features are often obscured by the noise, making it challenging for both MLP and KAN to extract these features effectively. We hope these experiments provide valuable insights for future neural network research and encourage further investigations to overcome these challenges.

## 1 INTRODUCTION

Since its launch, Kolmogorov-Arnold networks (KAN)(Liu et al., 2024b) has garnered significant attention. These networks utilize the Kolmogorov-Arnold representation theorem, which posits that any multivariate continuous function can be expressed as a combination of continuous univariate functions and addition. Unlike conventional Multi-Layer Perceptron (MLP) networks, KANs incorporate learnable activation functions. According to (Liu et al., 2024b), this feature provides KANs with enhanced interpretability and accuracy over MLPs.

Noting the release of KAN 2.0 (Liu et al., 2024a), we will conduct all experiments using the latest pykan version (v0.2.6) to leverage its optimizations. KAN 2.0 introduced multiplication nodes, which significantly enhance the fitting of multivariate functions, especially those involving direct multiplication or division of independent variables. However, the improvement is minimal for the functions used in this paper, so lower versions of Pykan are also acceptable.

Numerous investigations into KAN applications have rapidly surfaced, covering areas such as smart energy grid optimization(Wang et al., 2024)(Tang et al., 2024), chemistry data analysis(Wang et al., 2024)(Li et al., 2024b), image classification(Cheon, 2024)(Teymoor Seydi, 2024)(Igali & Shamoi, 2024), deep function learning(Zhang, 2024), quantum architecture search(Kundu et al., 2024), medical image analysis and processing(Li et al., 2024a)(Chen et al., 2024), disease risk predictions(Dong, 2024), graph learning tasks(Kiamari et al., 2024)(Li, 2024)(Ghaith Altarabichi, 2024), asset pricing models(Wang & Singh, 2024), 3D object detection (in autonomous driving)(Lai et al., 2024), sentiment analysis(Lawan et al., 2024), and deep kernel learning(Zinage et al., 2024).

On the contrary, a growing body of research has highlighted the imperfections of KANs compared to MLPs. For example, (Zhang, 2024) and (Shen et al., 2024) noted the vulnerability of KANs to noise, indicating that even minor noise can lead to a significant rise in test loss. Additionally, (Tran et al., 2024) claimed that KANs do not outperform MLPs in highly complex datasets and require considerably more hardware resources. Furthermore, (Yu et al., 2024) noted that MLPs generally have higher accuracy than KANs across various standard machine learning tasks, with the exception of tasks involving symbolic formula representation.

{1}------------------------------------------------

Moreover, the performance of networks may be influenced by the regularity of functions, prompting this study to compare the performance of KAN and MLP, across distinct types of functions with varying degrees of regularity (or noisy functions).

In this investigation, we assess the performance of MLP and KAN in modeling irregular or noisy functions. To ensure fairness, we control the number of parameters and the amount of training data. Moreover, we investigate the influence of different optimizers on the accuracy of fitting specific functions. This research continues directly and naturally from our recent study on the efficacy of KANs in fitting noisy functions (Shen et al., 2024).

The structure of this paper is organized as follows: Section 2 provides an introduction to the Kolmogorov-Arnold Theorem and KANs, discussing their benefits and limitations, and enumerates the six types of functions. Section 3 evaluates the performance of MLP and KAN in approximating regular and irregular functions. Section 4 introduces noise to the previously utilized functions and continues the comparison between MLP and KAN. Finally, Section 5 summarizes the findings of our experiments.

## 2 KOMOGOROV-ARNOLD THEOREM AND KANs

The Kolmogorov-Arnold theorem pertains to expressing multivariable continuous functions. According to the theorem, any continuous function involving multiple variables can be expressed as a combination of continuous single-variable functions and addition (Kolmogorov, 1956) (Kolmogorov, 1957) (Arnold, 1957). Formally, it can be stated as:

**Theorem 1.** [Kolmogorov-Arnold Theorem] *Let  $f : [0, 1]^n \rightarrow \mathbb{R}$  be any multivariate continuous function, there exist continuous univariate functions  $\phi_i$  and  $\psi_{ij}$  such that:*

$$f(x_1, x_2, \dots, x_n) = \sum_{i=1}^{2n+1} \phi_i \left( \sum_{j=1}^n \psi_{ij}(x_j) \right). \quad (1)$$

Leveraging the Kolmogorov-Arnold theorem, KANs introduce a novel neural network architecture. Unlike traditional Multi-Layer Perceptrons (MLPs) which use fixed activation functions, KANs employ learnable activation functions. This methodology is theoretically advantageous in enhancing the adaptability of KANs across different datasets and applications.

Unfortunately, Theorem 1 was originally proven non-constructively, lacking a constructive proof initially. In 2009, (Braun & Griebel, 2009) presented a constructive proof for this theorem. Nevertheless, challenges emerge when handling functions exhibiting irregular patterns, which mathematical analysis typically categorizes into at least five distinct types. Table 1 outlines these types along with detailed examples.

## 3 COMPARISON ON IRREGULAR FUNCTIONS

Here, we offer some explanations for the initial five types mentioned in Table 1. We compare these functions using multiple sets of KAN and MLP networks that have similar parameter counts. The number of parameters for each network are presented in Table 2. It is worth noting that the features of these functions are typically not available as prior knowledge in real applications, so we do not specifically select the corresponding network structure for each function. We implement the L-BFGS optimizer for these functions as it shows better performance in small-scale training. For functions with singularities or coherent oscillations, which might need more training samples and iterations, we also investigate the Adam optimizer’s capability.

### 3.1 REGULAR FUNCTIONS

First, consider the functions exhibiting strong regularity. Such functions are continuous and differentiable at all points, similar to  $f_1$  and  $f_2$ . We reconstruct these two functions using two sets of MLP and KAN networks that have comparable parameters but are trained with different sample sizes. The outcomes are displayed in Figure 1. It can be observed that for this category of functions, KAN outperforms MLP.

{2}------------------------------------------------

Table 1: Several Types of Functions and Their Examples

| Regular | Smooth |  |
|-|-|-|
|  | $f_1(x) = x^2$<br><img alt="Graph of f1(x) = x^2, a parabola opening upwards." data-bbox="412 259 502 322" src="7134640a6b83a95a2363e73eb40db625_img.jpg"/> | $f_2(x) = e^x$<br><img alt="Graph of f2(x) = e^x, an exponential growth curve." data-bbox="671 259 758 322" src="58603603a09f8039c5655106b565270b_img.jpg"/> |
| Irregular | Continuous everywhere except at points of non-differentiability |  |
|  | $f_3(x) =  x $<br><img alt="Graph of f3(x) =  x , a V-shaped absolute value function." data-bbox="412 399 502 461" src="6ef3174b70f11286c2cf8a5a99f24dcd_img.jpg"/> | $f_4(x) = 1 - \sqrt{ x }$<br><img alt="Graph of f4(x) = 1 - sqrt( x ), a function with a sharp peak at x=0." data-bbox="671 399 758 461" src="ef6d8e628b84f3bae23028ca81dc4137_img.jpg"/> |
|  | Jump |  |
|  | $f_5(x) = \begin{cases} 1, &  x  < 0.5 \\ 0, & \text{other} \end{cases}$<br><img alt="Graph of f5(x), a rectangular pulse function." data-bbox="412 567 502 630" src="3bae1bcc392f05587e8e6308e261cf1b_img.jpg"/> | $f_6(x) = \begin{cases} 1 - 4x^2, &  x  < 0.5 \\ 1, & \text{other} \end{cases}$<br><img alt="Graph of f6(x), a function with a parabolic jump at x=0." data-bbox="671 567 758 630" src="66e3a041bcfd6281d3213b7251ea6783_img.jpg"/> |
|  | Singular |  |
|  | $f_7(x) = \frac{1}{x}$<br><img alt="Graph of f7(x) = 1/x, a hyperbola with two branches." data-bbox="412 707 502 770" src="c7a969668e90097edc72ec33d78abc57_img.jpg"/> | $f_8(x) = \frac{1}{1-x^2} - 1$<br><img alt="Graph of f8(x), a function with two vertical asymptotes." data-bbox="671 707 758 770" src="28427b42cd814d26ba0bf545dae8915f_img.jpg"/> |
|  | Coherent oscillation |  |
|  | $f_9(x) = \cos(\frac{1}{x})$<br><img alt="Graph of f9(x) = cos(1/x), showing high-frequency oscillations near x=0." data-bbox="412 861 502 923" src="5c5f5078892f6fb881f26162bfb108d6_img.jpg"/> | $f_{10}(x) = \cos(\frac{2\pi}{1-x^2})$<br><img alt="Graph of f10(x), showing high-frequency oscillations near x=1 and x=-1." data-bbox="671 861 758 923" src="6e1d2ada686ca45711fb5546525c99bd_img.jpg"/> |
| Noisy | Noisy |  |
|  | $y = x + n(x)$ , where $n(x)$ denotes additive noise.<br><img alt="Graph of y = x + n(x), a noisy linear function." data-bbox="507 1007 671 1092" src="b3c978fb01a8871fb8488773078b9f64_img.jpg"/> |  |

Table 2: The number of parameters for each KAN and MLP network.

| MLP |  | KAN |  |  |  |
|-|-|-|-|-|-|
| width | parameter | width | grid | k | number of parameters |
| [1,39,1] | 118 | [1,5,1] | 3 | 3 | 120 |
| [1,79,1] | 238 | [1,10,1] | 3 | 3 | 240 |

{3}------------------------------------------------

Table 3: Time consumption of L-BFGS and Adam optimizers in fitting functions  $f_7$  and  $f_8$  using MLP and KAN

| Function | Network | Optimizer | Time(s) |
|-|-|-|-|
| $f_7$ | MLP | L-BFGS | 8.3069 |
| $f_7$ | MLP | Adam | 4.3064 |
| $f_7$ | KAN | L-BFGS | 588.8074 |
| $f_7$ | KAN | Adam | 38.4595 |
| $f_8$ | MLP | L-BFGS | 8.6801 |
| $f_8$ | MLP | Adam | 4.8102 |
| $f_8$ | KAN | L-BFGS | 359.7498 |
| $f_8$ | KAN | Adam | 39.4296 |

Table 4: Time consumption of L-BFGS and Adam optimizers in fitting functions  $f_9$  and  $f_{10}$  using MLP and KAN

| Function | Network | Optimizer | Time(s) |
|-|-|-|-|
| $f_9$ | MLP | L-BFGS | 8.4784 |
| $f_9$ | MLP | Adam | 4.5564 |
| $f_9$ | KAN | L-BFGS | 237.6449 |
| $f_9$ | KAN | Adam | 38.9890 |
| $f_{10}$ | MLP | L-BFGS | 5.8473 |
| $f_{10}$ | MLP | Adam | 4.8208 |
| $f_{10}$ | KAN | L-BFGS | 347.3375 |
| $f_{10}$ | KAN | Adam | 38.6431 |

### 3.2 CONTINUOUS FUNCTIONS WITH POINTS WHERE DERIVATIVES DO NOT EXIST

The functions  $f_3$  and  $f_4$  serve as prime examples of this category. They maintain continuity at all points, while they are non-differentiable at  $x = 0$ .

The outcomes are illustrated in Figure 2. For these particular functions, the KAN’s performance is worse than the MLP’s. Despite the MLP’s slower convergence, it eventually reaches a lower test loss. Additionally, it can be noted that amplifying the training sample size marginally enhances the performance of both networks. However, in the vicinity of the non-differentiable point, the MLP shows more significant improvement than the KAN. More visually, it is evident that the fitting performance of MLP and KAN around  $x = 0$  (the non-differentiable point) is approximately the same. Yet, with a larger training sample size, the MLP demonstrates superior fitting performance near  $x = 0$  compared to the KAN.

### 3.3 FUNCTIONS WITH JUMP

The examples of this category include  $f_5$  and  $f_6$ . These functions have jump discontinuities at  $x = \pm 0.5$ , where the function values abruptly change between 0 and 1. The experimental outcomes for these functions are depicted in Figure 3. Results show that the MLP outperforms the KAN. Moreover, expanding the training dataset size can enhance both networks’ performance to a certain extent. Nevertheless, KAN consistently fails to match the performance of MLP.

### 3.4 FUNCTIONS WITH SINGULARITIES

Functions possessing singularities display distinct behaviors, marked by a rapid change rate as they near these points, with their first derivative tending towards infinity at the singularity. Additionally, for any chosen continuous interval that omits these singularities, the functions remain continuous and differentiable across the interval.

To avoid division by zero and guarantee clear fitting results, the ranges of the functions  $f_7$  and  $f_8$  are limited to  $[0.001, 1]$  and  $[-0.999, 0.999]$ , respectively. We examined the effects of the training sample size, the number of Epochs, and the selection of optimizer on the fitting performance. As illustrated in Figure 4, simply enlarging the sample size by itself does not substantially enhance the performance when recovering  $f_7$  and  $f_8$ .

Pykan offers two optional optimizers: Adam and L-BFGS. As depicted in Figure 5, L-BFGS achieves faster convergence, whereas networks with Adam converges to a lower test loss. However, it is crucial to recognize that with a fixed learning rate, the improvement from this strategy is naturally constrained. As illustrated in Table 3, for an identical number of epochs, the training duration of the network with the L-BFGS optimizer is frequently several times greater than with the Adam optimizer.

Drawing from earlier experiments and findings, the fitting tests will utilize the Adam optimizer set with a learning rate of 0.01. As shown in Figure 6, KAN outperformed MLP in terms of fitting functions with singularities at the same number of epochs.

{4}------------------------------------------------

### 3.5 FUNCTIONS WITH COHERENT OSCILLATIONS

A unique type of functional singularity, labeled as 'coherent oscillatory singularity,' is exemplified by functions  $f_9$  and  $f_{10}$ . These functions display 'unreachable points' (e.g.,  $x = 0$  for  $f_9$ ), where as the function approaches these points, its values oscillate increasingly rapidly, intersecting the x-axis infinitely often.

In the experimental phase, taking a similar approach as described in section D. As shown in Figure 4 and 7, an increase in the sampling rate did not markedly enhance fitting accuracy. Particularly, within the KAN network framework, the optimizer L-BFGS outperformed Adam for function  $f_9$ , while for function  $f_{10}$ , Adam showed superior results. On the other hand, when fitting both functions with an MLP, Adam consistently performed better than L-BFGS.

In a similar manner, Table 4 demonstrates that employing the L-BFGS optimizer during the fitting process usually resulted in an additional increase in computational time. Figure 8 demonstrates that KAN consistently surpasses MLP when comparing performance over the same number of epochs.

## 4 COMPARISON ON NOISY FUNCTIONS

In the following, we discuss the roles of noisy functions. We introduce noise to functions previously discussed and proceed to evaluate the performance of MLP and KAN. According to the conclusions drawn in the preceding section, we will classify the functions into three categories: regular functions, functions with localized irregularities, and functions with severe discontinuities.

### 4.1 ADDING NOISE TO REGULAR FUNCTIONS

We introduce noise to functions exhibiting strong regularity, and subsequently fit these noisy data using KAN and MLP. The experimental outcomes are depicted in Figure 9. Our observations indicate that KAN achieves a lower test loss with low noise levels but performs worse under high noise conditions. When comparing the function fitting effect, the conclusion remains consistent: MLP shows better performance with minor noise interference, but KAN rapidly outperforms MLP as the training sample size increases.

### 4.2 ADDING NOISE TO FUNCTIONS WITH LOCALIZED IRREGULARITIES

Noise is subsequently added to  $f_3$ ,  $f_4$ ,  $f_5$ , and  $f_6$ . The experimental findings are shown in Figure 10. For  $f_3$  and  $f_4$ , the network can still capture some of the irregular features with a larger training sample. However, for  $f_5$  and  $f_6$ , both KAN and MLP perform poorly. The networks still have difficulty identifying the jump discontinuities, even with an increased sample size.

### 4.3 ADDING NOISE TO FUNCTIONS WITH SEVERE DISCONTINUITIES

Figure 11 shows that KAN's performance surpasses that of MLP when noise is added to functions with singularities or coherent oscillation. Interestingly, from the perspective of test loss, the impact of noise on fitting such functions is minimal. This phenomenon highlights the ineffectiveness of strategies relying solely on increased sampling rates in such instances.

## 5 CONCLUSION

In this study, we evaluate the effectiveness of KAN and MLP in approximating irregular or noisy functions. Our analysis concentrates on two main factors: the relative performance of KAN and MLP in fitting functions with different types according to regularity, and their ability to handle noise during the fitting process.

Firstly, as identified in (Shen et al., 2024) and additionally explored in this study, raising the sampling rate is a potent method to enhance the fitting performance of functions  $f_1 - f_6$ . Particularly, this strategy shows greater advantages when handling noisy data versus clean data. Nevertheless, the improvement in the fitting accuracy for functions with low regularity ( $f_7 - f_{10}$ ) is minimal, irrespective of the presence of noise.

{5}------------------------------------------------

Secondly, we also compared the fitting performance under varying Epochs from two distinct perspectives: convergence speed and stabilized test loss. KAN exhibits a faster convergence rate than MLP across all tested functions. However, MLP outperforms KAN on test functions  $f_3 - f_6$  on stabilized test loss.

Thirdly, via experimental analysis (fitting  $f_7 - f_{10}$ ), it was observed that Adam exceeded L-BFGS in performance for both networks in every instance, except for function  $f_9$ . Notably, when fitting function  $f_9$  with the KAN, L-BFGS demonstrated better results than Adam.

At last, when dealing with noisy functions, KAN exhibits superior performance over MLP for regular functions or functions with severe discontinuities. Conversely, for functions with localized irregularities, MLP outperforms KAN.

![Figure 1: Recover f1 and f2 independently using KAN and MLP under different training sample sizes. (a) f1: Test Loss vs Training Samples. (b) f2: Test Loss vs Training Samples.](ac99eff233b8fe51d30f499e7413c345_img.jpg)

Figure 1 consists of two line plots, (a) and (b), showing Test Loss versus Training Samples for functions  $f_1$  and  $f_2$  respectively. Both plots compare KAN (grid=3, k=3) and MLP (various architectures: [1.5, 1], [1.39, 1], [1.10, 1], [1.79, 1]). In both cases, KAN shows a significantly lower test loss than MLP as the number of training samples increases from 1000 to 5000. The test loss for KAN stabilizes around 0.0005, while MLP's test loss remains higher, around 0.0015 to 0.0020.

Figure 1: Recover f1 and f2 independently using KAN and MLP under different training sample sizes. (a) f1: Test Loss vs Training Samples. (b) f2: Test Loss vs Training Samples.

Figure 1: Recover  $f_1$  and  $f_2$  independently using KAN and MLP under different training sample sizes.

![Figure 2: Recover f3 and f4 independently using KAN and MLP. The figure is a 4x4 grid of plots. Rows 1 and 2 show training loss over epochs and samples for f3 and f4. Rows 3 and 4 show real values and predictions for f3 and f4 with different training sample sizes (50 and 5000).](e1dda754c2c88a8ad0b968aea4fc0786_img.jpg)

Figure 2 is a 4x4 grid of plots. The first two rows (a-d) show training loss over epochs or training samples for functions  $f_3$  and  $f_4$ . The last two rows (e-l) show the real values and predictions for  $f_3$  and  $f_4$  using MLP and KAN with different training sample sizes (50 and 5000). In the bottom rows, the real values are shown as green lines, and the predictions are shown as red lines. For  $f_3$  (a V-shape), both MLP and KAN perform well. For  $f_4$  (a bell-shaped curve), MLP's predictions (blue line) are smoother and follow the real values (green line) more closely than KAN's predictions (red line), especially with 50 training samples.

Figure 2: Recover f3 and f4 independently using KAN and MLP. The figure is a 4x4 grid of plots. Rows 1 and 2 show training loss over epochs and samples for f3 and f4. Rows 3 and 4 show real values and predictions for f3 and f4 with different training sample sizes (50 and 5000).

Figure 2: Recover  $f_3$  and  $f_4$  independently using KAN and MLP.

{6}------------------------------------------------

324  
325  
326  
327  
328  
329  
330  
331  
332  
333  
334  
335  
336  
337  
338  
339  
340  
341  
342  
343  
344  
345  
346  
347  
348  
349  
350  
351  
352  
353  
354  
355  
356  
357  
358  
359  
360  
361  
362  
363  
364  
365  
366  
367  
368  
369  
370  
371  
372  
373  
374  
375  
376  
377

![Figure 3: Recover f5 and f6 independently using KAN and MLP. The figure consists of eight subplots arranged in a 2x4 grid. The top row (a-d) shows results for f5, and the bottom row (e-h) shows results for f6. Columns 1 and 2 show training loss over epochs (0-50) for KAN (grid=3, k=3) and MLP (1,79,1). Columns 3 and 4 show real values and predictions over the domain [-1, 1] for training samples of 50 and 5000 respectively. In all cases, KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line).](c0843c6d138705289960d9f53a6e72a1_img.jpg)

Figure 3: Recover f5 and f6 independently using KAN and MLP. The figure consists of eight subplots arranged in a 2x4 grid. The top row (a-d) shows results for f5, and the bottom row (e-h) shows results for f6. Columns 1 and 2 show training loss over epochs (0-50) for KAN (grid=3, k=3) and MLP (1,79,1). Columns 3 and 4 show real values and predictions over the domain [-1, 1] for training samples of 50 and 5000 respectively. In all cases, KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line).

Figure 3: Recover  $f_5$  and  $f_6$  independently using KAN and MLP.

![Figure 4: Recover f7, f8, f9 and f10 independently using KAN and MLP with optimizer Adam or L-BFGS, 2000 Epochs. The figure consists of four subplots: (a) f7, Adam for MLP and KAN; (b) f8, Adam for MLP and KAN; (c) f9, Adam for MLP, L-BFGS for KAN; (d) f10, Adam for MLP and KAN. Each plot shows training loss over 1000 epochs. KAN (red dashed line) consistently achieves lower loss than MLP (blue dashed line).](c64e9e9f3b0b828a5f6ac70441877764_img.jpg)

Figure 4: Recover f7, f8, f9 and f10 independently using KAN and MLP with optimizer Adam or L-BFGS, 2000 Epochs. The figure consists of four subplots: (a) f7, Adam for MLP and KAN; (b) f8, Adam for MLP and KAN; (c) f9, Adam for MLP, L-BFGS for KAN; (d) f10, Adam for MLP and KAN. Each plot shows training loss over 1000 epochs. KAN (red dashed line) consistently achieves lower loss than MLP (blue dashed line).

Figure 4: Recover  $f_7$ ,  $f_8$ ,  $f_9$  and  $f_{10}$  independently using KAN and MLP with optimizer Adam or L-BFGS, 2000 Epochs

![Figure 5: The variation of test loss with the increasing number of epochs when recovering f7 and f8 independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP with optimizer L-BFGS and Adam, learning rate=0.01. The figure consists of four subplots: (a) MLP, f7; (b) KAN, f7; (c) MLP, f8; (d) KAN, f8. Each plot shows test loss over 1500 epochs. KAN (red dashed line) converges to a lower test loss faster than MLP (blue dashed line).](01da0d212fb571933f10f96556157745_img.jpg)

Figure 5: The variation of test loss with the increasing number of epochs when recovering f7 and f8 independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP with optimizer L-BFGS and Adam, learning rate=0.01. The figure consists of four subplots: (a) MLP, f7; (b) KAN, f7; (c) MLP, f8; (d) KAN, f8. Each plot shows test loss over 1500 epochs. KAN (red dashed line) converges to a lower test loss faster than MLP (blue dashed line).

Figure 5: The variation of test loss with the increasing number of epochs when recovering  $f_7$  and  $f_8$  independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP with optimizer L-BFGS and Adam, learning rate=0.01

![Figure 6: Recover f7, (x in [0.001, 1]) and f8, (x in [-0.999, 0.999]) independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP both with optimizer Adam, same or different epochs. The figure consists of four subplots: (a) f7, MLP; (b) f7, KAN; (c) f8, MLP; (d) f8, KAN. Each plot shows training loss over 1.5 epochs. KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line).](023b142f90e1253702ac88b18380d3ec_img.jpg)

Figure 6: Recover f7, (x in [0.001, 1]) and f8, (x in [-0.999, 0.999]) independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP both with optimizer Adam, same or different epochs. The figure consists of four subplots: (a) f7, MLP; (b) f7, KAN; (c) f8, MLP; (d) f8, KAN. Each plot shows training loss over 1.5 epochs. KAN (red dashed line) fits the target function (green squares) much better than MLP (blue dashed line).

Figure 6: Recover  $f_7$ , ( $x \in [0.001, 1]$ ) and  $f_8$ , ( $x \in [-0.999, 0.999]$ ) independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP both with optimizer Adam, same or different epochs

{7}------------------------------------------------

378  
379  
380  
381  
382  
383  
384  
385  
386  
387  
388  
389  
390  
391  
392  
393  
394  
395  
396  
397  
398  
399  
400  
401  
402  
403  
404  
405  
406  
407  
408  
409  
410  
411  
412  
413  
414  
415  
416  
417  
418  
419  
420  
421  
422  
423  
424  
425  
426  
427  
428  
429  
430  
431

![Figure 7: Variation of test loss with increasing number of epochs when recovering f9 and f10. Subplots (a) MLP, f9; (b) KAN, f9; (c) MLP, f10; (d) KAN, f10. Each plot shows test loss (y-axis) vs. training samples (x-axis, 0 to 2000). Adam (blue line) and L-BFGS (orange line) optimizers are compared. In all cases, L-BFGS achieves lower test loss faster than Adam.](a71911ad87414271aeb190e0eebcb989_img.jpg)

Figure 7: Variation of test loss with increasing number of epochs when recovering f9 and f10. Subplots (a) MLP, f9; (b) KAN, f9; (c) MLP, f10; (d) KAN, f10. Each plot shows test loss (y-axis) vs. training samples (x-axis, 0 to 2000). Adam (blue line) and L-BFGS (orange line) optimizers are compared. In all cases, L-BFGS achieves lower test loss faster than Adam.

Figure 7: The variation of test loss with the increasing number of epochs when recovering  $f_9$  and  $f_{10}$  independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP with optimizer L-BFGS and Adam, learning rate=0.01

![Figure 8: Recover f9 and f10 independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP. Subplots (a) f9, MLP (Adam); (b) f9, KAN (L-BFGS); (c) f10, MLP (Adam); (d) f10, KAN (Adam). Each plot shows the function value (y-axis, -1.0 to 2.0) vs. x (x-axis, -1.0 to 1.0). Real data points are shown as green dots, and predictions are shown as lines. KAN (b, d) shows better fit to the real data than MLP (a, c).](ecb25d766719ce041cf4cc390791a098_img.jpg)

Figure 8: Recover f9 and f10 independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP. Subplots (a) f9, MLP (Adam); (b) f9, KAN (L-BFGS); (c) f10, MLP (Adam); (d) f10, KAN (Adam). Each plot shows the function value (y-axis, -1.0 to 2.0) vs. x (x-axis, -1.0 to 1.0). Real data points are shown as green dots, and predictions are shown as lines. KAN (b, d) shows better fit to the real data than MLP (a, c).

Figure 8: Recover  $f_9$ , ( $x \in [-0.999, 0.999]$ ) and  $f_{10}$ , ( $x \in [-0.999, 0.999]$ ) independently using [1,10,1] KAN (grid=3,k=3) and [1,79,1] MLP with same or different optimizers and epochs

![Figure 9: Recover f1 and f2 with noise independently using KAN and MLP. The figure is a 4x3 grid of plots. Row 1: Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP, f1 with noise. Row 2: Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP, f2 with noise. Row 3: Original data, noisy data and predictions, f1 with noise, training samples = 50. Row 4: Original data, noisy data and predictions, f2 with noise, training samples = 5000 (500 plotted for clarity). Columns: (a) Original data, noisy data and predictions, f1 with noise, training samples = 50; (b) Original data, noisy data and predictions of MLP, f1 with noise, training samples = 5000 (500 plotted for clarity); (c) Original data, noisy data and predictions of KAN, f1 with noise, training samples = 5000 (500 plotted for clarity).](27b22513fc27a0ff5f230b062ad3112f_img.jpg)

Figure 9: Recover f1 and f2 with noise independently using KAN and MLP. The figure is a 4x3 grid of plots. Row 1: Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP, f1 with noise. Row 2: Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP, f2 with noise. Row 3: Original data, noisy data and predictions, f1 with noise, training samples = 50. Row 4: Original data, noisy data and predictions, f2 with noise, training samples = 5000 (500 plotted for clarity). Columns: (a) Original data, noisy data and predictions, f1 with noise, training samples = 50; (b) Original data, noisy data and predictions of MLP, f1 with noise, training samples = 5000 (500 plotted for clarity); (c) Original data, noisy data and predictions of KAN, f1 with noise, training samples = 5000 (500 plotted for clarity).

Figure 9: Recover  $f_1$  and  $f_2$  with noise independently using KAN and MLP.

{8}------------------------------------------------

432  
433  
434  
435  
436  
437  
438  
439  
440  
441  
442  
443  
444  
445  
446  
447  
448  
449  
450  
451  
452  
453  
454  
455  
456  
457  
458  
459  
460  
461  
462  
463  
464  
465  
466  
467  
468  
469  
470  
471  
472  
473  
474  
475  
476  
477  
478  
479  
480  
481  
482  
483  
484  
485

![Figure 10: Recover f3, f4, f5 and f6 with noise independently using KAN and MLP. The figure consists of 13 subplots (a-m) arranged in a 4x4 grid. Subplots (a-d) show training loss vs. training samples for various noise levels (0, 2, 4, 10) for functions f3, f4, f5, and f6 respectively. Subplots (e-n) show original data (black line), noisy data (green dots), and predictions (red line) for functions f3, f4, f5, and f6. Each subplot compares KAN (grid=3, k=3) and MLP (grid=3, k=3) models. The noise level is 10 for all functions. Training samples used are 50, 500, 1000, 2000, 3000, 4000, and 5000. For clarity, only 500 noisy data points are plotted in (f-n).](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

(a) Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP,  $f_3$  with noise

(b) Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP,  $f_4$  with noise

(c) Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP,  $f_5$  with noise

(d) Various training samples, [1,5,1] KAN (grid=3, k=3) and [1,39,1] MLP,  $f_6$  with noise

(e) Original data, noisy data and predictions,  $f_3$  with noise, training samples = 50

(f) Original data, noisy data and predictions of MLP,  $f_3$  with noise, training samples = 5000 (500 plotted for clarity)

(g) Original data, noisy data and predictions of KAN,  $f_3$  with noise, training samples = 5000 (500 plotted for clarity)

(h) Original data, noisy data and predictions,  $f_4$  with noise, training samples = 50

(i) Original data, noisy data and predictions of MLP,  $f_4$  with noise, training samples = 5000 (500 plotted for clarity)

(j) Original data, noisy data and predictions of KAN,  $f_4$  with noise, training samples = 5000 (500 plotted for clarity)

(k) Original data, noisy data and predictions,  $f_5$  with noise, training samples = 50

(l) Original data, noisy data and predictions,  $f_5$  with noise, training samples = 5000 (500 plotted for clarity)

(m) Original data, noisy data and predictions,  $f_6$  with noise, training samples = 50

(n) Original data, noisy data and predictions,  $f_6$  with noise, training samples = 5000 (500 plotted for clarity)

Figure 10: Recover f3, f4, f5 and f6 with noise independently using KAN and MLP. The figure consists of 13 subplots (a-m) arranged in a 4x4 grid. Subplots (a-d) show training loss vs. training samples for various noise levels (0, 2, 4, 10) for functions f3, f4, f5, and f6 respectively. Subplots (e-n) show original data (black line), noisy data (green dots), and predictions (red line) for functions f3, f4, f5, and f6. Each subplot compares KAN (grid=3, k=3) and MLP (grid=3, k=3) models. The noise level is 10 for all functions. Training samples used are 50, 500, 1000, 2000, 3000, 4000, and 5000. For clarity, only 500 noisy data points are plotted in (f-n).

Figure 10: Recover  $f_3$ ,  $f_4$ ,  $f_5$  and  $f_6$  with noise independently using KAN and MLP.

{9}------------------------------------------------

486  
487  
488  
489  
490  
491  
492  
493  
494  
495  
496  
497  
498  
499  
500  
501  
502  
503  
504  
505  
506  
507  
508  
509  
510  
511  
512  
513  
514  
515  
516  
517  
518  
519  
520  
521  
522  
523  
524  
525  
526  
527  
528  
529  
530  
531  
532  
533  
534  
535  
536  
537  
538  
539

![Figure 11: Recover f7, f8, f9 and f10 with noise independently using KAN and MLP. The figure consists of 12 subplots arranged in a 4x3 grid. The first column (a, d, g, j) shows 'Train Loss' vs 'Training Samples' for various models (origin (KAN), origin (MLP), SNR=10 (KAN), SNR=10 (MLP), SNR=0 (KAN), SNR=0 (MLP), SNR=4 (KAN), SNR=4 (MLP)). The second column (b, e, h, k) shows 'Original Data' (black dashed line) and 'Noisy Data' (green dots) with predictions from an MLP model. The third column (c, f, i, l) shows 'Original Data' (black dashed line) and 'Noisy Data' (green dots) with predictions from a KAN model. Rows correspond to functions f7, f8, f9, and f10 respectively. Subplots (b, c, e, f, h, i, k, l) include a zoomed-in view of the noisy data points around the original function curve.](7801d00a216dc4dc8a7d210dcb5fe3c5_img.jpg)

(a) Various training samples,  $[1,10,1]$  KAN (grid=3,  $k=3$ ) and  $[1,79,1]$  MLP,  $f_7$  with noise

(b) Original data, noisy data and predictions,  $f_7$  with noise, MLP

(c) Original data, noisy data and predictions,  $f_7$  with noise, KAN

(d) Various training samples,  $[1,10,1]$  KAN (grid=3,  $k=3$ ) and  $[1,79,1]$  MLP,  $f_8$  with noise

(e) Original data, noisy data and predictions,  $f_8$  with noise, MLP

(f) Original data, noisy data and predictions,  $f_8$  with noise, KAN

(g) Various training samples,  $[1,10,1]$  KAN (grid=3,  $k=3$ , Opt=L-BFGS) and  $[1,79,1]$  MLP (Opt=Adam),  $f_9$  with noise

(h) Original data, noisy data and predictions,  $f_9$  with noise, MLP

(i) Original data, noisy data and predictions,  $f_9$  with noise, KAN

(j) Various training samples,  $[1,10,1]$  KAN (grid=3,  $k=3$ ) and  $[1,79,1]$  MLP,  $f_{10}$  with noise

(k) Original data, noisy data and predictions,  $f_{10}$  with noise, MLP

(l) Original data, noisy data and predictions,  $f_{10}$  with noise, KAN

Figure 11: Recover f7, f8, f9 and f10 with noise independently using KAN and MLP. The figure consists of 12 subplots arranged in a 4x3 grid. The first column (a, d, g, j) shows 'Train Loss' vs 'Training Samples' for various models (origin (KAN), origin (MLP), SNR=10 (KAN), SNR=10 (MLP), SNR=0 (KAN), SNR=0 (MLP), SNR=4 (KAN), SNR=4 (MLP)). The second column (b, e, h, k) shows 'Original Data' (black dashed line) and 'Noisy Data' (green dots) with predictions from an MLP model. The third column (c, f, i, l) shows 'Original Data' (black dashed line) and 'Noisy Data' (green dots) with predictions from a KAN model. Rows correspond to functions f7, f8, f9, and f10 respectively. Subplots (b, c, e, f, h, i, k, l) include a zoomed-in view of the noisy data points around the original function curve.

Figure 11: Recover  $f_7$ ,  $f_8$ ,  $f_9$  and  $f_{10}$  with noise independently using KAN and MLP.

 Rest of paper (reference and Appendix) is removed.