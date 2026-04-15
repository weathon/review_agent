# Estimating Post-Synaptic Effects for Online Training of Feed-Forward SNNs

- Decision: Reject
- Scores: 5, 3, 5

## Abstract
Facilitating online learning in spiking neural networks (SNNs) is a key step in developing event-based models that can adapt to changing environments and learn from continuous data streams in real-time. Although forward-mode differentiation enables online learning, its computational requirements restrict scalability. This is typically addressed through approximations that limit learning in deep models. In this study, we propose Online Training with Postsynaptic Estimates (OTPE) for training feed-forward SNNs, which approximates Real-Time Recurrent Learning (RTRL) by incorporating temporal dynamics not captured by current approximations, such as Online Training Through Time (OTTT) and Online Spatio-Temporal Learning (OSTL). We show improved scaling for multi-layer networks using a novel approximation of temporal effects on the subsequent layer's activity. This approximation incurs minimal overhead in the time and space complexity compared to similar algorithms, and the calculation of temporal effects remains local to each layer. We characterize the learning performance of our proposed algorithms on multiple SNN model configurations for rate-based and time-based encoding. OTPE exhibits the highest directional alignment to exact gradients, calculated with backpropagation through time (BPTT), in deep networks and, on time-based encoding, outperforms other approximate methods. We also observe sizeable gains in average performance over similar algorithms in offline training of Spiking Heidelberg Digits with equivalent hyper-parameters (OTTT/OSTL – 70.5%; OTPE – 75.2%; BPTT – 78.1%).

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a new online training method OTPE for spiking neural networks. It incorporates temporal dynamics not captured by existing online methods such as OSTL and OTTT while maintaining similar time and space complexity. Experiments on synthetic datasets and SHD with feedforward networks demonstrate improved performance and better alignment of gradient angles with BPTT.

### Strengths
1. This paper considers facilitating online learning in SNNs, which is important for biological plausibility and neuromorphic hardware.

2. Experiments demonstrate large improvements on gradient angle alignment and better performance than existing methods.

### Weaknesses
1. The presentation can be improved. It is not quite clear what is the major component in the methodology that leads to the large improvement over existing methods. From Section 3, OTPE is very similar to OSTL and there is no explicit description of the key difference. From Section 3.1, AOTPE differs from OTTT only in that AOTPE uses a running weighted average of surrogate derivatives. Why this small change can almost double the cosine similarity given the similar time and space complexity? Is there more formal theoretical justification? From the derivation, OTPE also makes many approximations, why these approximations have less effect than OSTL/OTTT? Besides, what does the term “postsynaptic estimates” mean?

2. Scalability is mentioned multiple times in the paper. Are there more large-scale (datasets, network, etc.) results?

3. In the abstract, “rate-based and time-based encoding” is mentioned. However, there is no corresponding part in the paper.

### Questions
See weakness.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the author presents an Online Training algorithm with postsynaptic estimates (OTPE). OTPE leverages the preservation of multiple temporal spike outputs to achieve a more precise gradient. Their experiments demonstrate that, compared to similar algorithms, OTPE achieves better alignment with gradients by BPTT.

### Strengths
The authors introduce a new approximation RTRL algorithm called OTPE that captures richer temporal effects compared to previous online training methods. OTPE algorithm achieve  gradients highly aligned between BPTT method and effectively reduce the time and space complexity.

### Weaknesses
1. The OTTT algorithm was experimented on several datasets such as CIFAR10 and CIFAR100. In comparison, the paper only presented comparative experiments on SHD, which may lead to insufficient persuasive power of the experiments. Therefore, it may be necessary to conduct experiments on a more diverse range of datasets to fully validate the effectiveness of OTPE.
2. The presentation of the Approximate OTPE section in the paper seems unclear as it does not clearly demonstrate the mathematical approximation made by OTPE and Approximate OTPE.
3. The paper claims in the abstract that "This approximation incurs minimal overhead in the time and space complexity compared to similar algorithms." However, the theoretical analysis does not demonstrate the advantages, and there is a lack of corresponding experimental comparisons to support this claim.
4. The LIF neurons of the article seem to be missing $ s_t^{l-1}\cdot \theta $, so the membrane potential stays the state of decay. If the membrane potential update formula is modified to $s_t^l=H(U_{t}^l-V_{th}), U_t^l=\lambda U_{t-1}^l+ s_t^{l-1}\cdot \theta -V_{th}\cdot s_t^l$, then the OTPE's $ \frac{\partial U_t^l}{\partial \theta^{l-1}}=\frac{\partial U_t^l}{\partial s_t^{l-1}}\frac{\partial s_t^{l-1}}{\partial \theta^{l-1}} $ of the second term on the right-hand side What does it mean? Does this also mean that it is possible to compute the expression of the left form directly, without the need to use the chain rule, as in the article?
5. In the second paragraph of section 3, the expression $$\frac{\partial s(t)_i^l}{w_{ij}^l} $$, does the subscript "i" in the numerator refer to the current time step? Could you explain the meanings of the variables and improve the wording accordingly? Moreover, should the parameter "w" be denoted as $\theta$ based on the preceding text?

### Questions
See the weakness.

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper presents a novel method for the online training of feed-forward Spiking Neural Networks (SNNs) that incorporates temporal information from the previous and current spikes into a postsynaptic estimate of the Real-Time Recurrent Learning (RTRL). Previous implementations of online SNN learning (such as Online Training Through Time (OTTT) or Online Spatio-Temporal Learning (OSTL)) fails to account for the temporal information of previous spikes leading to reduced accuracy of the SNN. Using Online Training with Postsynaptic Estimates (OTPE) or its approximation, gradient estimates exhibited more alignment to the exact gradients generated from Backpropagation through time (BPTT) compared to previous approaches.

### Strengths
-	Good background explanation, especially with explaining the mathematics of previous methods in comparison to OTPE.
-	Table 2 clearly demonstrates how OTPE performs at a closer online accuracy level to BPTT offline accuracy for SNNs at different depths and widths compared to other approximation training methods.
-	Analysis was clear and transparent. Analysis was able to differentiate between actual results and faulty results due to the methodology. In the last paragraph before Section 5, the author says “The performance difference across OTPE, its approximations, OSTL, and OTTT is more apparent for T-Randman and SHD. Although we observed higher test accuracy for online learning on SHD than for offline learning, we attribute this to our hyperparameter search (see Appendix A.2).” Author also acknowledges intermediate/incomplete results in this paragraph when it’s said “OTPE does not appear to have reached peak validation accuracy even after training on 10,000 mini-batches.”

### Weaknesses
-	The F-OTPE method should’ve been tested in multiple application-specific SNN datasets such as Auditory N-MNIST, DVS-Speech, or Spiking MNIST Audio Dataset. It’s hard to tell whether the results presented for F-OTPE are valid and can be reproducible since the method was only tested on one dataset. 
-	Online training of OTPE only occurred on one dataset throughout the paper. If this is supposed to be a preferred method for online training, there should be more tests demonstrating the improved performance of OTPE in online SNN training using a variety of datasets such as Auditory N-MNIST, DVS-Speech, or Spiking MNIST Audio Dataset. The author can also try testing using image classification data (instead of audio) by using N-CARS, DvsGesture, or N-MNIST.

### Questions
-	[Introduction, 1st paragraph] The authors say that “However, BPTT is unsuitable for online learning (Kaiser et al., 2020; Bohnstingl et al., 2022)”. Throughout the paper, I see them using gradients from BPTT as the ground truth for their approach. Can you explain this discrepancy?
-	In Figure 2, I observe that the proposed approach is as good as BPTT, then why won’t I just use BPTT? What advantage is the OTPE giving?
-	[Introduction, 2nd paragraph] “To address these limitations ...” All the references are bunched up so it is hard to understand which technique points to which reference.
-	Many of the figures were unclear or hard to read, e.g., all the line plots in Fig 2a are bunched up, for Figure 5 the markers are obfuscating the color of the line.
-	Please explain F-OTPE more thoroughly in the Section 3.2. Elaborate on the line “the loss can be calcualted similarly to how cross-entropy loss is calculated in offline training.” Also include the equations involved for F-OTPE and how it relates to normal OTPE. While the explanation in the paper was fine, it will be better explained through a visual of the mathematics.
-	Why wasn’t F-OTPE tested along with OTPE and approx OTPE in the offline training scenarios presented in Figures 2, 3, 4, 5? 
-	How will OTPE, approx OTPE, F-OTPE perform in a different application-specific dataset? You can stick to audio datasets by using Auditory N-MNIST, DVS-Speech, Spiking MNIST Audio Dataset, etc. Or you can expand to image classification datasets by using N-CARS, DvsGesture, N-MNIST, etc.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
3 good
