# Test-Time Accuracy-Cost Control in Neural Simulators via Recurrent-Depth

- Decision: Accept (Poster)
- Scores: 8, 2, 8, 4

## Abstract
Accuracy-cost trade-offs are a fundamental aspect of scientific computing. Classical numerical methods inherently offer such a trade-off: increasing resolution, order, or precision typically yields more accurate solutions at higher computational cost. We introduce \textbf{Recurrent-Depth Simulator} (\textbf{RecurrSim}) an architecture-agnostic framework that enables explicit test-time control over accuracy-cost trade-offs in neural simulators without requiring retraining or architectural redesign. By setting the number of recurrent iterations $K$, users can generate fast, less-accurate simulations for exploratory runs or real-time control loops, or increase $K$ for more-accurate simulations in critical applications or offline studies. We demonstrate RecurrSim's effectiveness across fluid dynamics benchmarks (Burgers, Korteweg-De Vries, Kuramoto-Sivashinsky), achieving physically faithful simulations over long horizons even in low-compute settings. On high-dimensional 3D compressible Navier-Stokes simulations with 262k points, a 0.8B parameter RecurrFNO outperforms 1.6B parameter baselines while using 13.5\% less training memory. RecurrSim consistently delivers superior accuracy-cost trade-offs compared to alternative adaptive-compute models, including Deep Equilibrium and diffusion-based approaches. We further validate broad architectural compatibility: RecurrViT reduces error accumulation by 90\% compared to standard Vision Transformers on Active Matter, while RecurrUPT matches UPT performance on ShapeNet-Car using 44\% fewer parameters.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduce a new procedure for training any block of neural architectures when learning solutions of PDEs. This procedure consists of incorporating recurrent calls to the block, whose number is controlled by a parameter $K$. The parameter $K$ changes during the training to make the obtained reccurent network able to learn the solution for any $K$, with the intuition that the approximation will be more accurate for high $K$ than for low $K$. As a result, it is possible to tune the accuracy-cost trade-off at test time by toggling $K$. The approach is validated on several benchmark and for several underlying neural architectures.

### Strengths
- The paper is well written, easy and pleasant to follow
- The idea is simple, original, and represents a clever approach for adding an inductive bias towards physical solvers in the obtained neural network together with controlling the cost-accuracy trade-off. 
- The approach is thoroughly validated on small to large scale physical learning problem, and its applicability to different existing SOTA architectures is demonstrated (RecurrFNO, RecurrVIT, RecurrUPT).

### Weaknesses
- The benchmark would benefit from a more systematic evaluation of UPT, ViT and FNO, i.e applying those tree models and their Recurr variants to all three high dimensional datasets.
- The high dimensional benchmark lacks a study on the effect of $K$.

### Questions
Are there practical limitations that prevented the authors to apply UPT, ViT and FNO and their Recurr variants to all three high dimensional datasets?

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
This paper proposes an architecture-agnostic framework, the Recurrent-Depth Simulator. During the training phase, the framework randomly samples the number of recurrent iterations K from a distribution and optimizes using truncated backpropagation; during the test phase, users can explicitly specify the number of iterations K to trade off between computational cost and simulation accuracy. The authors validate this framework across multiple datasets, including Burgers, Korteweg-de Vries (KdV), Kuramoto-Sivashinsky (KS), high-dimensional Compressible Navier-Stokes (CNS), Active Matter, and ShapeNet-Car. The method is compared against other adaptive-compute models, such as FNO-DEQ, ACDM, and PDE-Refiner, as well as standard architectures such as FNO, ViT, and UPT. The paper concludes that RecurrSim offers a superior accuracy-cost trade-off curve compared to baselines. On the high-dimensional CNS task, a lower-parameter RecurrFNO variant outperforms a higher-parameter FNO baseline while also reducing training memory.

### Strengths
- The framework's core contribution is providing explicit test-time control, allowing users to flexibly trade computational cost for accuracy by adjusting the number of iterations. Compared to baselines, this method offers a smoother, more predictable trade-off curve, avoiding the early saturation or erratic behavior seen in alternatives.
- The framework achieves excellent parameter efficiency through weight-sharing, enabling it to match or exceed larger baseline models with significantly fewer parameters and lower training memory consumption.
- The method is a plug-and-play, architecture-agnostic framework, and its generality has been validated across diverse backbones, including FNO, ViT, and UPT

### Weaknesses
- The core mechanism of this work,a recurrent-depth block trained with truncated backpropagation, is conceptually very similar to a standard Recurrent Neural Network (RNN), making the contribution potentially incremental as it applies existing techniques to a new domain.

- The paper is lacking in visual comparisons. The authors didn't provide corresponding visualizations for baselines like FNO-DEQ or ACDM, making it difficult to visually assess differences in physical fidelity. Also, none of the cases are provided with range.

- The paper suffers from several typographical errors and unclear phrasings. In particular, the descriptions of some experimental setups (e.g., Section 4.3 ) are brief, which may create difficulties for readers attempting to reproduce the results.

### Questions
See weaknesses

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the Recurrent-Depth Simulator (RecurrSim), an architecture-agnostic framework designed to give neural simulators explicit, test-time control over their accuracy-cost trade-off. This capability is standard in classical numerical methods but largely absent in modern deep learning-based simulators.
The core idea is to replace a fixed-depth network with a recurrent block that is iterated a user-specified number of times (K) at inference. The model is trained by sampling K from a distribution and using truncated backpropagation-through-depth to maintain a fixed memory footprint. The authors demonstrate RecurrSim's effectiveness on a wide range of benchmarks and show that it can be applied to various backbones, consistently outperforming standard architectures and other adaptive-compute models.

### Strengths
1. Sufficient Novelty: The paper addresses a critical and practical problem in scientific machine learning. While the core mechanism (using recurrent iterations for an accuracy-cost trade-off) has been explored in other domains like computer vision and natural language processing, this paper's novelty lies in Its application and validation for the neural simulator domain, where this feature is a standard expectation from classical solvers but has been a major missing piece for deep learning methods.
2. Methodological Simplicity and Generality: The RecurrSim framework is "plug-and-play." It requires minimal code changes to a standard architecture – like LoRa methods. The paper strongly supports its "architecture-agnostic" claim by successfully applying it to FNO, ViT, and UPT.
3. Strong comparison with other baselines and in a wide range of PDE problems.
4. Scalability and Efficiency: The results on 3D CNS are impressive (a 0.8B param RecurrFNO outperforms a 1.6B param FNO with 13.5% less training memory). 
5. High-Quality Presentation: The paper is exceptionally clear, well-structured, and easy to follow. The appendices provide strong justifications for design choices.

### Weaknesses
1. Lack of a Dedicated Reproducibility Section recommended in author guidelines (although appendix provides enough)
2. Insufficient Justification for truncated backpropagation-through-depth: The authors propose truncated backpropagation-through-depth to bound memory. However, they fail to discuss or compare this to gradient checkpointing, a standard alternative. Gradient checkpointing would compute the exact full-depth gradient (trading compute for memory) instead of the approximate gradient from truncated backpropagation-through-depth. The paper provides no justification for why an approximate gradient is sufficient or preferable.

### Questions
1. Minor comment Line 1215 Optimization typo. Can you correct?
2. You justify using truncated backpropagation-through-depth as a way to bound memory, which provides an approximate gradient. Could you elaborate on why this was chosen over gradient checkpointing, a standard alternative that computes the exact full-depth gradient by trading compute for memory?
3. The ICLR guidelines strongly encourage a dedicated 'Reproducibility Statement' paragraph to help reviewers locate the relevant details. While the appendices provide excellent, comprehensive details for reproducibility, this specific statement is missing. Would the authors be willing to add this paragraph in the final version to improve clarity for future readers?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose a very simple framework for controlling, at test-time, the accuracy/speed of a neural simulator model, without requiring retraining or architecture adaptations. They show that this technique can be incorporated into a variety of architectures.

### Strengths
The proposed framework is easily used in multiple different architectures, and that flexibility is a strong point.

No additional custom losses or tricks are required, and the authors provide a simple explanation of the algorithm, making adoption simple.

The authors demonstrate improved performance over baselines with reduced compute/parameter counts.

### Weaknesses
The authors say that repeated applications of the recurrent block lead encourage the recurrent block to contract toward a fixed point — what is the justification for this claim? Is there any theoretical proof that the recurrent blocks do indeed converge toward a fixed point?

I would like to see some ablations on the initial latent distribution. The authors claim that the choice “primarily affects early iterations”. The authors also show that the early iterations are the ones that lead to the largest reduction in L2 error and are the most “important” in this sense, and so it would be interesting to see whether the choice of the initial latent distribution makes a big different in terms of overall performance of this method.

### Questions
Could the authors more clearly distinguish their method from DEQ, which also repeatedly applies a function (here the recurrent block) and converges to a fixed point, with the number of function applications being controllable to achieve a desired accuracy? 

How was the recurrent iteration distribution chosen? It would be interesting to see how changing this distribution changes the performance of the model. 

The authors show in the top of Figure 2 that performance saturates relatively quickly with the number of recurrent steps K, with the earliest steps leading to the largest reduction in L2 error. However, for memory purposes, the authors use a fixed backpropagation window where only the last B steps are backpropagated through, with the earlier steps being treated as constant. Would it not make more sense to backpropagate through the earliest recurrent layers, given that the earliest ones are the the ones that lead to the largest reduction in L2 error?

### Soundness
2

### Presentation
3

### Contribution
2
