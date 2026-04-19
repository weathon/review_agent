# Guaranteed Approximation Bounds for Mixed-Precision Neural Operators

- Decision: Accept (poster)
- Scores: 6, 8, 5, 8

## Abstract
Neural operators, such as Fourier Neural Operators (FNO), form a principled approach for learning solution operators for partial differential equations (PDE) and other mappings between function spaces. However, many real-world problems require high-resolution training data, and the training time and limited GPU memory pose big barriers. One solution is to train neural operators in mixed precision to reduce the memory requirement and increase training speed. However, existing mixed-precision training techniques are designed for standard neural networks, and we find that their direct application to FNO leads to numerical overflow and poor memory efficiency. Further, at first glance, it may appear that mixed precision in FNO will lead to drastic accuracy degradation since reducing the precision of the Fourier transform yields poor results in classical numerical solvers. We show that this is not the case; in fact, we prove that reducing the precision in FNO still guarantees a good approximation bound, when done in a targeted manner. Specifically, we build on the intuition that neural operator learning inherently induces an approximation error, arising from discretizing the infinite-dimensional ground-truth input function, implying that training in full precision is not needed. We formalize this intuition by rigorously characterizing the approximation and precision errors of FNO and bounding these errors for general input functions. We prove that the precision error is asymptotically comparable to the approximation error. Based on this, we design a simple method to optimize the memory-intensive half-precision tensor contractions by greedily finding the optimal contraction order. Through extensive experiments on different state-of-the-art neural operators, datasets, and GPUs, we demonstrate that our approach reduces GPU memory usage by up to 50% and improves throughput by 58% with little or no reduction in accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper intends to reduce the cost of training and inference of FNO. The method used here include training by mixed-precision to reduce memory and time cost. The mixed-precision training technique proposed here is specialized for FNO, where bounds for precision error and approximation error are theoretically analyzed and guaranteed. The experiment results show that the proposed method greatly improves the speed and throughput of FNO while not sacrificing accuracy.

### Strengths
**Originality:** The novelty of this paper lies in leveraging the analysis of precision error in Fourier transform, which successfully reduced the computational cost and maintained the accuracy of FNO.

**Quality:** The quality of the paper is high in both theoretical analysis and experiment conduction.

**Clarity:** The paper delivers its idea in a quite clear and detailed way.

**Significance:** For people who intend to have a large scale of implementations of FNO, this work can be very interesting to them.

### Weaknesses
The major doubt for this paper is regarding the significance. While it is dedicated to improving FNO, an important architecture of neural operators, FNO is not the only one and arguably the best one. If FNO is yet to be a dominant model for large scale applications of neural operators, the significance of this work is limited.

### Questions
None.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors demonstrated mixed-precision training of FNOs on GPUs.  They derived theoretic bounds on the effect of rounding error, improved AMP to support complex arithmetic, addressed an instability by introducing a pre-activation thah, and conducted experiments on practical datasets.

### Strengths
- Application is of significance.
- Well written. 
- Code released.

### Weaknesses
> It would be helpful to include an illustration of empirical characterization of rounding error in direct comparison with the theoretical bound scaling law.   
> Mixed-precision with FP8 has been proposed, how does the method fare with FP8?

### Questions
See minor requests above.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper the authors provide a way to apply mixed-precision training to Fourier Neural Operators. The authors point out that while some previous approaches have used mixed-precision training with FNO, they have done so only on the real-valued parameters (i.e, the linear weights, and biases) and not on the FFT/DFT that is applied to the input in each layer. 

They first show that mixed-precision training should not cause extra error in the training due to the Fourier transform, since the discretization error from FFT→ DFT is orders of magnitude bigger than the DFT→ half-precision DFT. 

For the application of mixed-precision on complex tensors (the output after an FFT/DFT application) the authors apply mixed-precision by converting each “tensors to real” (this is something that the authors don’t elaborate upon enough). 

The authors show through their experiments, that they are indeed able to train various FNO based architectures with low-memory and hence are also able to increase the throughput of their runs while only taking a less than 1 percent hit in the performance as compared to non-mixed-precision baselines.

The authors also point to the use of a tanh based pre-activation that helps in mitigating the mixed-precision based overflow that usually occurs when trained FNO based architectures.

### Strengths
The paper is well-written and easy to follow.

The experimental results of the paper are impressive. The authors are able to increase the training throughput for navier stokes $1.41$ times, by achieving an almost 50 percent memory reduction. 

The use of tanh activation as a pre-activation to the neural operator block is a good technique to reduce the overflow issue, and an important finding. 

The authors provide a theoretical proof for why the error for the mixed-precision training would be negligible when compared to the discretization error for DFT, while the results there are pretty standard its a good addition to have.

### Weaknesses
I think that the primary methodology as to how the mixed-precision is applied to the Fourier Kernel is not clear. From the looks of it the mixed-precision approximation is applied to the weights in the complex domain (that are used in the kernal operator).

However, this is something that seems to be different from what they say in the introduction, where they mention that they enable some mixed precision in the entire FNO block (which I assumed will try to also enabled mixed precision in the DFT algorithm). 

From the looks of it the primary contribution seems to be the addition of tanh, and the application of mixed precision in the complex domain by treating the real and the imaginary components as two distinct real tensors. Together, the overall contribution does not seem very novel in of itself.

### Questions
The authors mention that neural operators are discretization convergent, however have not cited any relevant work around it. While empirically we know that for some FNO based architecture we can get zero shot super-resolution, are there any works that prove it? In general, adding relevant citation to that claim would be useful.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a mixed-precision training method for neural operators, focusing on Fourier Neural Operators (FNO) used in solving partial differential equations and other mappings between function spaces. The paper discusses the challenges of high-resolution training data, limited GPU memory, and long training times in the context of neural operators. It emphasizes the need for mixed-precision training to mitigate these issues. The paper demonstrates that, contrary to our expectations, mixed-precision training in FNO does not lead to significant accuracy degradation. It presents rigorous theoretical characterization of approximation and precision errors in FNO, highlighting that the precision error is comparable to the approximation error. The paper introduces a method for optimizing memory-intensive tensor contractions using mixed precision, reducing GPU memory usage by up to 50% and improving throughput by 58% while maintaining accuracy.

 These findings have the potential to advance the efficiency and scalability of neural operators in various downstream applications.

### Strengths
The strengths of the paper are as follows:

1. Mixed-Precision Training for Neural Operators: The paper introduces the first mixed-precision training routine tailored specifically for neural operators. This novel approach optimizes the memory-intensive tensor contraction operations in the spectral domain and incorporates the use of tanh pre-activations to address numerical instability. 

2. Theoretical Results; The paper provides a strong theoretical foundation for its work by characterizing the precision and discretization errors of the FNO block. It demonstrates that these errors are comparable and proves that, when executed correctly, mixed-precision training of neural operators does not lead to significant performance degradation. 

3. Experimental Evaluation: The paper conducts thorough empirical validation of its mixed-precision training approach on three state-of-the-art neural operators (TFNO, GINO, and SFNO) across four different datasets and GPUs. The results indicate that the method significantly reduces memory usage (using half the memory) and increases training throughput by up to 58% across various GPUs, all while maintaining high accuracy with less than 0.1% reduction. 

4. Open-Source Code: The paper provides an efficient implementation of its approach in PyTorch, making it open-source and providing all the necessary data to reproduce the results.

### Weaknesses
While the method of using tanh pre-activation before each FFT seems to avoid numerical instability, it would have been better if some theoretical justification was given (even for simplistic cases). I believe similar theory should have been provided for the learning rate schedule. However, the paper indeed contributes significantly in theoretical aspects (in Section 3) by characterizing the precision and discretization errors of the FNO block and showing that these errors are comparable. Given the strong theoretical contributions, the above is not significant. Moreover, a strong ablation study for tanh is provided in Appendix B.5

### Questions
The work is indeed very interested and provides a strong contribution to the research community. My questions are:

1. Why is it that the standard mixed precision training used for training ConvNets and ViTs is ineffective for training Fourier Neural Operators?
2. Similarly, why is is that the common solutions (loss scaling, gradient clipping, normalization, delaying updates) fail to address the numerical instability of mixed-precision FNO?

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent
