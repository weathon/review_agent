# Lightweight Detection of Silent Data Corruption in Distributed Deep Learning

- Avg Score: 3.00
- Decision: Reject
- Scores: 0, 4, 6, 2

## Abstract
Reliable detection of silent data corruption (SDC), such as bit-flip errors, is critical in large-scale neural network training, as undetected hardware faults can silently propagate and severely degrade model performance. We introduce a lightweight detection method integrated directly before collective communication steps, enabling localization of faulty devices with minimal runtime overhead. Our approach combines statistical modeling of gradient norms with divergence-based criteria to improve robustness. Experiments on large-scale training workloads, including LLaMA2-7B, show that our detector successfully identifies the vast majority of high-order bit-flip faults in bfloat16 while incurring only a very small computational overhead, offering a strong balance between detection accuracy and efficiency.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
5

### Summary
This work presents a statistical method for lightweight detection of SDCs (Silent Data Corruption) in distributed neural network training. This is done by checking for multimodality in the distribution of NN gradients - if the gradient distribution is unimodal, the test passes and SDCs are assumed to not exist. In the event of the test showing multimodality, the Wasserstein distance at the pivot point of the set of gradients (the distance between these halves) is used to check for significant distribution distance. This is then used to flag the presence of an SDC. This approach is asymptotically computationally cheap, scaling as O(mlogm). 

Having worked on SDCs for quite a while in the hardware space, I have several questions about this approach:
1) There has been extensive work on low-cost algorithm-based fault tolerance in the hardware space. Algorithmic checksums have been shown theoretically to achieve O(1) correction of single bit errors in linear matrix multiplications [1], and that same paper illustrates prior art dating back to the  1980s implementing these checksums in high-performance computing. More recent work has achieved very high resilience using simpler statistical tests on neuron statistics in reinforcement learning training [2]. How is this work distinct and more importantly scalable, given the greater sophistication of Wasserstein distance computations and the folding test? Citations of prior art and contrasts with prior art are needed.
2) On that note, the current work pays attention to all-reduce operations and communications between work units in training, hence focusing on gradient values. Ozen et al [3] have achieved extremely low cost resilience in the forward pass using linear algorithmic checksums, and by setting neuron values to zero in inference [4]. Given that this may also work for forward pass computations in training, one can regard the forward pass as secure? If so, are the bitflips injected here into backward pass computations? I do not see clear locations of injection in the training flow, and contrast with prior work.
3) How does this approach contrast with other hardware-focused fault tolerance mechanisms such as resilience-aware scheduling, redundancy, algorithmic noise tolerance? The related work section ignores a great deal of prior art.
4) Lastly, I do not see a clear noise model, or an accurate one. I quote, "We inject 300 faults in total, uniformly across three flip types (100 each), where type-1/2/3 means flipping the 1st/2nd/3rd bit of the IEEE 754 exponent, respectively." Ozen et al have shown in [5] that bit error rates are as low as 1e-4 or 1e-5 in real operation, and this is distributed across the entire IEEE754 word, uniformly - not just the most significant bits. As such, how does this noise model, simply flipping the most significant exponent bits of the word, in any way test the sensitivity of this framework? Surely randomized injection corresponding to a probability of bit error would be more realistic?

[1]  Franklin T. Luk and Haesun Park "An Analysis Of Algorithm-Based Fault Tolerance Techniques", Proc. SPIE 0696, Advanced Algorithms and Architectures for Signal Processing I, (4 April 1986); https://doi.org/10.1117/12.936896 

[2] C. Amarnath, M. Mejri, J. Isenberg and A. Chatterjee, "Error Resilient Online Reinforcement Learning Using Adaptive Statistical Checks," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 44, no. 8, pp. 3112-3125, Aug. 2025, doi: 10.1109/TCAD.2025.3529820

[3] Ozen, E., Orailoglu, A. Low-Cost Error Detection in Deep Neural Network Accelerators with Linear Algorithmic Checksums. J Electron Test 36, 703–718 (2020). https://doi.org/10.1007/s10836-020-05920-2

[4] Elbruz Ozen and Alex Orailoglu. 2020. Just say zero: containing critical bit-error propagation in deep neural networks with anomalous feature suppression. In Proceedings of the 39th International Conference on Computer-Aided Design (ICCAD '20). Association for Computing Machinery, New York, NY, USA, Article 75, 1–9. https://doi.org/10.1145/3400302.3415680

[5] E. Ozen and A. Orailoglu, "Boosting Bit-Error Resilience of DNN Accelerators Through Median Feature Selection," in IEEE Transactions on Computer-Aided Design of Integrated Circuits and Systems, vol. 39, no. 11, pp. 3250-3262, Nov. 2020, doi: 10.1109/TCAD.2020.3012209. keywords: {Feature extraction;Machine learning;Resilience;Hardware;Task analysis;Error analysis;Neural networks;Approximate computing;fault tolerance;neural network hardware;neural networks},

### Strengths
The paper has a strong statistical theory part and the idea of testing for SDCs using distribution consistency metrics and modality is a very sound one. Furthermore, I feel that its segments on realistic architectures and communication during training and its focus on distributed systems renders it very promising, should it be able to rectify the issues above. Prior art, dating back as it does to the 1980s, does not adequately focus on this new space, despite being highly applicable there today. Overall it is a very promising space and a very solid theoretical foundation.

### Weaknesses
The weaknesses of the paper are twofold:
1) Due to the large amount of prior art, some of which I have cited above, it does not adequately contrast itself to that prior work and does not cite it. Given that linear algorithmic checksums and ECCs form the basis of bit error correction systems in production today, and the hardware community has been working on neural network errors for a while, I think a literature survey is badly needed in this paper to distinguish it from the state of the art - and indeed, the state of prior art.
2)  Its experimental work section is flawed. There is no real hardware-consistent emulation of faults, and bit error injection is conducted in by selecting the largest possible bits and flipping them. There is no real need for a statistical threshold and clever statistical mechanics if a simple hard threshold on the gradients at extreme values of 1e5 will detect errors - the experiments therefore do not show the need for the technique. The authors need to compare against prior art, yes, but also more importantly need to adequately stress test their own technique! Without that this paper absolutely should not fly.

### Questions
I would like to know more about the experimental setup and the number format. While the injection into highest exponent bits seems excessive to me, did your gradient values reflect that? Were encountered gradient values within an order or two of magnitude of that? If so then I will retract my earlier comment in Paper Weaknesses, but to me it feels like the values injected for bitflips are too extreme to even test the method, let alone compare to prior art.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents a lightweight statistical pipeline designed to detect Silent Data Corruption (SDC), specifically high-order bit-flip errors, during distributed deep learning training. The detector is integrated into the communication layer. It mainly checks gradient norms before collective operations to enable localization of faulty devices and prevent error propagation with minimal overhead. The detection method includes logarithmic transformation of gradient norms to enhance separation between clusters, folding test for unimodality check, and if multimodality is detected, the Wasserstein distance is computed between the separated modes. Experiments on LLaMA2-7B training showed 99.0% TPR in detecting injected high-order bit-flips with no FPs.

### Strengths
1. The detector successfully identifies the vast majority of high-order bit-flips, resulting in a 99% TPR in LLaMA2-7B training with zero false positives.
2. The method is highly lightweight, designed to integrate into large-scale training with minimal runtime overhead. It incurred a measured runtime overhead of approximately 0.46% per iteration during LLaMA2-7B training.

### Weaknesses
1. The experimental evaluation was limited to a single model architecture, the LLaMA2 - 7B model. Detailed validation across diverse set of architectures and training scales is required to fully assess the applicability of the proposed approach. 
2. The detection method is specifically designed to identify high-order bit-flips and the impact of other bit-flips is not discussed in detail.
3. The decision threshold for the Wasserstein distance was determined empirically in the proposed technique. 
4. The entire foundation of the method is based on the assumption that SDC manifests as high-order bit-flips that destroy the normal distribution and create heavy tails.

### Questions
1. Why specifically LLaMA2-7B is selected for evaluation?
2. How does the proposed technique perform for models other than LLaMA2-7B?
3. Why 300 faults are injected across only three flip types? Is there any significance behind this number and configuration or not?
4. Why just the three bit locations? Is this configuration sufficient to cover all the cases highlighted in Section 2.3?
5. It is stated that the decision threshold for the Wasserstein distance was determined empirically in the proposed technique. How can it be defined for new models? Through simulations? Is it dependent on the settings used for simulation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work contributes to methods for recognizing difficult to diagnose hardware failures in large-scale deep learning training workloads, focusing on instances of silent data corruption (SDC), such as random bit-flip errors. Such corruption can result in catastrophic problems depending on where it occurs, leading to "NaN" or "inf" values at bfloat16 precision, the most common floating point precision for training in modern architectures at scale. Their method adds checks before collective operations to identify corrupted tensors, isolating instances of SDC. Combining multimodal detection of gradient norms with distance-based divergence metrics, this method achieves 99% detection accuracy on Llama2 7B training workloads with no false positives and minimal runtime overhead. Finally, the authors have built an adapter for Huawei Ascend devices such that their approach requires no high-level training code changes for those devices.

### Strengths
- The contribution in this work is clearly very relevant to those in the deep learning community interested in training at scale or models that result from training at scale. Along these lines, the importance of a solution to the outlined problem is also fairly well established in this work.
- Given the assumptions in this work (and assuming for a moment that they are true regarding e.g. unimodality or weak multimodality of most distributions within modern models), the proposed methodology is sensible and logically laid out. In addition, the proposed method is fairly simple and easy to understand.
- The experimental results are mostly compelling and that there were no false positives is especially helpful towards lending the method in this work credence. In addition, that the runtime overhead is only ~2% is compelling.

### Weaknesses
- While not necessarily the fault of this work (due to the nature of this problem, where corruption can be very difficult to detect and isolate), I had some difficulty finding data outlining just how often errors that this method would apply to actually occur. The issues described by this work are almost certainly critical to deal with for large-scale workloads, but that lingering question of “how often do hardware failures occur?” hurts the potential impact of this work.
- Related to the above point, while just ~2% runtime overhead is a strong result, there is no easy way to get a sense of how significant of a tradeoff is being made between that runtime overhead compared to resolving specific data-corruption errors. Another way of looking at this is that a work like this would benefit significantly from something akin to an impact estimate. So something to the effect of: “the proposed method results in 2% runtime overhead but resolves errors that would on average result in X% overhead for checkpoint rollbacks.”
- The number of injected faults is not actually that convincing. Ideally the dataset used for experiments would be larger to get a better sense of how this method would perform at very large scales. Or, if the number of injected faults is actually realistic for pre-training large models, that could probably be better explained.
- Similar to the above point, I remain a little confused as to how the number of false positives (or lack thereof) were measured. More accurately, what is this out of? As in, how many samples were introduced? I assume this means that some normal, control inputs were used? If so, this should be mentioned when outlining the main experiment in this work.

### Questions
See the comments above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper looks at the issue of silent data corruptions, and introduces a lightweight statistical check to detect errors. The check operates on the gradient norms, and various metrics are included to measure the error probability. Fault injection is performed to measure robustness.

### Strengths
+ Important problem domain. The multimodal/algorithmic part is interesting

### Weaknesses
- Unfortunately, not much that is presented is new or novel. The statistical approach has been studied previously (citations below), LLMs have been studied previously in the context of SDCs, number formats are not studied in this paper (but are very important), and multi-modal has also been studied. To that end, while this is a very important domain, I was not able to extract the novelty of the technique, and the related work really was lacking.
- The evaluation has a very low number of error injections. What is the statistical significance of 300 faults total, in a 7B model?
- How is the fault injector implemented? Is it a software FI tool? Or done in hardware? It seems to be a software tool like PyTorchFI/TensorFI/Ares; however, such tools are not always truly accurate at measuring real SDC rate, as lower order tools are better at capturing hardware propagation of bit flips.
- Many of the insights are not new: exponent bits being vulnerable; incorporating the use of the gradient; etc. Overall, it is hard to see what truly is different in this work, from many other works in the area.

### Questions
Please address the related work, and discuss why such a technique is lightweight, or should be employed in production beyond modern techniques.

Related work to explore:
- Demystifying the Resilience of Large Language Model Inference: An End-to-End Perspective (SC 2025)
- Hardware Sentinel: Protecting Software Applications from Hardware Silent Data Corruptions (ASPLOS 2025)
- https://www.opencompute.org/documents/sdc-in-ai-ocp-whitepaper-final-pdf
- Hardware Resilience Properties of Text-Guided Image Classifiers (NeurIPS 2023)
- GoldenEye: A Platform for Evaluating Emerging Numerical Data Formats in DNN Accelerators (DSN 2022)
- Demystifying the system vulnerability stack: Transient fault effects across the layers (ISCA 2021)
- Reliability Evaluation of Compressed Deep Learning Models (LASCAS 2020)
- FIdelity: Efficient Resilience Analysis Framework for Deep Learning Accelerators (2020)
- Analytical guarantees on numerical precision of deep neural networks (ICML 2017)

### Soundness
2

### Presentation
2

### Contribution
2
