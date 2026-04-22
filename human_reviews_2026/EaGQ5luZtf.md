# Light Differentiable Logic Gate Networks

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 4, 6, 8, 6

## Abstract
Differentiable logic gate networks (DLGNs) exhibit extraordinary efficiency at inference while sustaining competitive accuracy.
But vanishing gradients, discretization errors, and high training cost impede scaling these networks.
Even with dedicated parameter initialization schemes from subsequent works, increasing depth still harms accuracy.
We show that the root cause of these issues lies in the underlying parametrization of logic gate neurons themselves.
To overcome this issue, we propose a reparametrization that also shrinks the parameter size logarithmically in the number of inputs per gate.
For binary inputs, this already reduces the model size by 4x, speeds up the backward pass by up to 1.86x, and converges in 8.5x fewer training steps.
On top of that, we show that the accuracy on CIFAR-100 remains stable and sometimes superior to the original parametrization.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a new parameterization called input-wise parameterization (IWP) for Differentiable Logic Gate Networks (DLGNs), which converts the original parameterization (OP), where mixture-over-Boolean-functions parameterization is learned. IWP directly learns the numerical outcome of four input cases, i.e., (0,0), (0,1), (1,1), (1,0), leading to a 4x parameters reduction per gate. The authors claim that the new proposed parameterization can alleviate gradient vanishing and less computational costs. Experiments on CIFAR-100 show superior performance and faster computation of IWP compared to the original parameterization.

### Strengths
1. The analysis in discretization error, redundancy, and gradient stability of OP in Section 3 is insightful.
2. The manuscript is well-organized and easy to follow.

### Weaknesses
1. My major concern of the paper is the experiment. The authors only evaluate IWP on single CIFAR-100 datasets. I would expect the authors to include more diverse datasets, for example, MNIST, CIFAR-10, WMT’14, and possibly TinyImageNet. Experiments on more diverse datasets are needed to evaluate the proposed parameterization robustly. 
2. My second concern is that the improvement over OP is marginal. For example, in CIFAR-100 the authors show 2% improvement with 8x faster convergence. However, there is no breakthrough in representation power, i.e., the depth scaling behaviors are unchanged compared to the OP as seen in Figure 4. However, note that the overall test accuracy is 29% while very small CNNs like resnet-20 can have 70%+ accuracy on CIFAR-100. Hence, results would be more compelling if deeper models with IWP closed the gap to standard architectures, not just DLGNs with OP.

### Questions
1. In line 196-197, should the argmax is the pass-through gate G4 rather than G3?
2. The expression of Equation 9 is a bit confusing to me. The equation would be more self-contained with more details like $G(k,l) = \alpha_{0,0} E_{k,l,0,0} + ... $ and $E_{k,l,i,j} = \textbf{1}_{\\{(k,l)=(i,j)\\}}$.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces an input-wise parametrization (IWP) for differentiable logic gate networks that removes redundancy, stabilizes gradients, and reduces discretization error. It cuts parameters by 4×, speeds up backward pass up to 1.86×, and converges 8.5× faster while preserving or improving accuracy on CIFAR benchmarks.

### Strengths
The paper presents a highly efficient reparametrization that eliminates redundancy in logic gate networks and significantly stabilizes gradients. It is grounded in a strong theoretical analysis of vanishing gradients and discretization error. The method delivers substantial improvements in training speed and resource efficiency. It also enables deeper DLGNs to train reliably without performance collapse. Finally, the approach maintains hardware compatibility, making it practical for edge and FPGA deployment.

### Weaknesses
See Question Below

### Questions
1) Missing ablation studies on hyperparameters such as learning rate and optimizer choices. Since logic-gate networks can be highly sensitive to gradient flow and gating stability, how do different learning rates and optimizers affect the gate sharpening process and convergence behavior in IWP-based DLGNs? Additionally, how robust is the model to hyperparameter variation, and does IWP mitigate instability compared to the original parametrization (**OP**)?

2) The experiments are conducted only on CIFAR-10/100, which are datasets from the same image family and share similar structure. To test generalization of the proposed IWP method to diverse problem domains, would including simpler symbolic datasets (e.g., **MNIST**) or non-vision benchmarks provide stronger evidence that the approach generalizes beyond natural images and is not over-specialized to CIFAR distributions?

3) Results show that IWP scales effectively with depth under residual initialization (**RI**). However, the study appears limited to thermometer encoding and nearest-rounding. Could the authors kindly clarify whether the stability and discretization improvements of IWP are expected to extend to alternative binary encodings or rounding strategies, and whether similar behavior holds without residual biasing?

4) Since IWP's parameter count scales as $2^n$ for $n$-input gates, did the authors examine performance and memory implications for $n>2$? Additionally, are there any observed or expected stability issues or computational constraints as the logic arity increases, particularly in terms of gradient behavior, training efficiency, or numerical sensitivity?

### Soundness
3

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
3

### Summary
This paper proposes an input-wise parameterization (IWP) of logic gate networks with tailored initializations that mitigate vanishing gradient issues of  DLGNs when increasing depth. Specifically, the paper points out that in previous DLGN, since there are negation pairs of logic gate in each layer, initializing the weight of each gate independently will cause gradient cancellation during back propagation. To address this issue, the paper reparameterize the logic function with fewer independent components. Experimental results show that the proposed method achieve better performance than existing DLGNs when increasing network depth.

### Strengths
1. The paper is well written and easy to follow
2. The paper is well motivated, as addressing the vanishing gradient issue of deep DLGNs will improve their applicability for more challenging tasks
3. The analysis of the root cause of vanishing gradient issue in DLGN , and the solution proposed by the paper make intuitive sense and are all backed by solid proof.
4. Experimental results and ablation study are comprehensive and positively support the proposed design.

### Weaknesses
1. The paper conducts experiment only on CIFAR 100.

### Questions
1. It would be better if the paper add comparison experiment on more complex dataset like ImageNet 32

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
2

### Summary
The paper tackles the problem of vanishing gradients, discretization errors, and high training cost of Differentiable logic gate networks (DLGNs). The authors claimed to root cause the issues and proposed a reparametrization solution to resolve it. Redundant parameters of input grates are replaced while maintaining the representability. With binary inputs, it can achieve 4x smaller model size, 1.86x backward pass speedup and 8.5x fewer training steps to converge.

### Strengths
1. Less weight parameters compared w/ the original DLGN paper.

### Weaknesses
DLGN seems to have a low test accuracy, which makes it less appealing as a practically useful solution.

### Questions
It makes sense that light weight gate parameters help resolving the vanishing gradient issue. How is this  solution working on a larger data set like imagenet?

### Soundness
3

### Presentation
3

### Contribution
3
