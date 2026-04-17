# FedChill: Adaptive Temperature Scaling for Federated Learning in Heterogeneous Client Environments

- Decision: Reject
- Scores: 4, 4, 2

## Abstract
Federated Learning (FL) enables collaborative model training with data privacy but suffers in non-i.i.d. settings due to client drift, which degrades both global and local generalizability. Recent works show that clients can benefit from lower softmax temperatures for optimal local training. However, existing methods apply a uniform value across all participants, which may lead to suboptimal convergence and reduced generalization in non-i.i.d. client settings. We propose FedChill, a heterogeneity-aware strategy that adapts temperatures to each client. FedChill initializes temperatures using a heterogeneity score, quantifying local divergence from the global distribution, without exposing private data, and applies performance-aware decay to adjust temperatures dynamically during training. This enables more effective optimization under heterogeneous data while preserving training stability. Experiments on CIFAR-10, CIFAR-100, and SVHN show that FedChill consistently outperforms baselines, achieving up to 8.35\% higher global accuracy on CIFAR-100 with 50 clients, while using 2.26$\times$ fewer parameters than state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposed FedChill which is a heterogeneity-aware strategy that adapts temperatures to each client.
FedChill initializes temperatures using a heterogeneity score, quantifying local divergence from the global distribution, without exposing private data, and applies performance-aware decay to adjust temperatures dynamically during training. In addition to extensive experiments, this paper also provides a convergence argument under bounded temperature schedules.

### Strengths
The heterogeneity based temperature schedule is simple to compute, privacy-aware, and also integrates with FedAvg. Per-client exponential temperature initialization strategy can ensure each client starts with a temperature value tailored to the distribution and complexity of its local data, which is in this reviewr’s view a key advantage of proposed solution.
The provided results cover client counts, Dirichlet heterogeneity levels, and model capacities.

### Weaknesses
The heterogenity score is based on class-prior mismatch to an ideal global distribution. As such, it may not capture feature shift or concept drift  which limits the generality of this solution beyond label-skew scenarios. The stagnation trigger (patience threshold P, decay, Tmin/Tmax, s) is empirically tuned. It is important to consider testing stability under partial participation, noisy validation, and non-stationary.
Evaluations rely on Dirichlet partitions and private validation sets carved from the global test data. There is a lack of evaluation of real-world datasets, label noise, on-device budget constraints, calibration metrics, and comparisons to equally simple baselines, e.g., comparing with the per-client learning-rate scaling or temperature-as-regularizer in the loss only scenario.

### Questions
-How does FedChill perform when heterogeneity is mainly caused by feature shift or concept drift? 
-Can the heterogeneity score be calculated in a representation space without public data? If so, how may that affect privacy and cost? 
-Can you compare FedChill’s adaptive decay to simpler per-client learning-rate schedules or entropy-regularized losses that respond to the same stagnation signal to isolate the causal benefit of temperature? 
-How sensitive are outcomes to s, P ,Tmin, and Tmax? Can you provide details of variance across clients and ablate patience/decay choices to assess stability and fairness at scale?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper mitigate the client draft problem in federated leaning by a proposed heterogeneity-based template scaling approach. The insight is that client with higher heterogeneity should have a higher temperature (smoother prediction and slower learning rate). Comprehensive experiments show good and coherent empirical evidence that supports the claim. However, several design and implementation details are not entirely clear to me: (1) how is the ideal global distribution defined? Is there a privacy concern? (2) Why parameter size differs in Table 1, and (3) how temperature scaling really helps? I would recommend this paper if these questions are clearly answered.

### Strengths
1. The motivation of this work is well-presented and Figure 1 (left) makes a very convincing point to me. I do enjoy reading this paper and the overall rollout of the story.
2. The experiments are pretty standard.

### Weaknesses
**What is the core mechanism of FedChill?**

Upon deeper thought, I believe there are two parts that adaptive temperatures work: creating a smoother prediction distribution and functioning as an adaptive learning rate. I am particularly interested in the second part. As FedExp [1] has shown that scaling the global learning rate proportionally to the inverse of gradient diversity helps, Equation (2) shows a similar formula (where the temperature is controlled by Equations 6 and 7). I think an additional experiment showcasing that vanilla FedAvg + an adaptive learning rate (not temperature) would further strengthen this point and provide a stronger explanation for how it works.

**Inconsistent model sizes in Table 1.**

In Table 1, the model parameter sizes differ, which makes the comparison unclear. It also appears that FedChill only works with a fixed model size of 4M. Can the authors provide a fairer comparison using consistent model sizes across all methods? (This is a part that confuses me a lot and please clarify it if I read it wrong)

**Applying FedChill to other algorithms.**

How well does FedChill work with other algorithms such as FedProx, FedAlign, and FedMD? Is there a synergistic effect? Having this additional experiment would strengthen the paper's claims about the algorithm's generalizability.

[1] FedExP: Speeding Up Federated Averaging via Extrapolation, ICLR 2023

### Questions
Besides the questions in the weaknesses session, here are some of the additional questions.

1. How to interpret Figure 1 (right) properly. While L124-125 mentioned "higher hetegogeneity benefit from lower temperatures", I do not see the same pattern. Could the authors explain this to me again?

2. How is the "ideal global distribution", mentioned in L209, exactly being implemented? Do you assume a global uniform distribution?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents FedChill, a method that improves federated learning under non-IID client data by adapting each client’s softmax temperature. A new heterogeneity score is proposed to initialize temperatures based on how far a client’s label distribution diverges from the estimated global distribution. Then, an adaptive decay mechanism lowers the temperature when validation accuracy plateaus.

### Strengths
- Per-client adaptive temperature scaling is intuitive, incurs no communication overhead, and integrates directly into FedAvg.
- Demonstrates clear improvements in both global and personalized performance, particularly under high heterogeneity.

### Weaknesses
- Several typos and figure clarity problems make the paper harder to follow such as Figure 1 axis labeling.
- Accuracy curves suggest training may not have fully converged by 50 rounds; longer experiments or convergence diagnostics would strengthen claims.
- Performance may rely heavily on hand-tuned scaling factors and patience settings, without guidance for general applicability.

### Questions
- the manuscript contains typos and grammatical errors starting from the abstract.
- in Figure 1 (left), the x-axis representation should be revised since client indices are discrete and fractional values are misleading. Improving visualization clarity would help convey the motivation more effectively.
- Table 1 and related descriptions suggest that some models may not be fully converged within 50 rounds. Please include longer training runs or convergence plots to confirm stability.
- Performance may depend strongly on the scaling factor and patience threshold used for temperature decay.
- Please avoid using AI generated text in the manuscript and ensure the writing reflects original academic quality.

### Soundness
2

### Presentation
2

### Contribution
1
