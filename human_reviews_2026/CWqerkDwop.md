# Optimization-Based Defender via Coarse-To-Fine Tensor Network Representation

- Avg Score: 5.50
- Decision: Reject
- Scores: 8, 4, 4, 6

## Abstract
Deep neural networks are vulnerable to well-designed adversarial attacks. Although numerous defense strategies have been proposed, most are tailored to specific threats or datasets and thus struggle to generalize across diverse adversarial scenarios. In this paper, we propose Tensor Network Purification (TNP), a novel optimization-based defense technique built upon a specially designed tensor network decomposition algorithm. TNP depends neither on the pre-trained generative model nor the specific dataset, enabling robust generalization. To this end, the key challenge lies in relaxing Gaussian-noise assumptions of classical decompositions and accommodating the unknown perturbation distributions. Instead of imposing consistency by traditional objectives, TNP aims to reconstruct the latent clean example from its adversarially perturbed input. Specifically, TNP leverages progressive downsampling with a new adversarial objective that minimizes reconstruction error while suppressing the inadvertent restoration of the perturbations. Extensive experiments on CIFAR-10, CIFAR-100, and ImageNet show that TNP generalizes effectively across diverse norm threats, attack types, and datasets, delivering a versatile and promising defense.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces an AP method combined a coarse-to-fine TN incremental learning algorithm with an adversarial optimization objective to defend adversarial attacks by mitigating adversarial perturbations. It’s training-free and shows its generalizable performance in various models and datasets and compared with other SOTA level AT and diffusion-based AP methods.

### Strengths
1. This paper introduces a novel method with a TN and dexterously combined with an adversarial optimization objective which effectively address the gaps in AT, AP and TN, shows the SOTA level performance and it’s training free.
2. The theory is presented clearly, and the theory diagrams are also clear.
3. The experiments are concise and comprehensive with various models and datasets under various attacks. Providing some visual results for making more convinced. And the compared methods are up-to-date.

### Weaknesses
1. There’s no results w/o auxiliary variable in Figure 4. And there has comprehensive comparison in Table 14 but in appendix.
2. The author chose the hyperparameter $\eta$ in Table 15 and Table 16. However, due to the significant performance differences among the 0, 0.1 and 0.2, the experimental granularity is not sufficient to convince readers that 0.1 is the relatively better choice.
3. The experiment has not been run multiple times, and there is no report of the mean and standard deviation of the results.

### Questions
1. Could you add a result w/o auxiliary variable in Figure 4, or integrate the comprehensive comparison into the main text rather than appendix.
2. Could you conduct a more granular search within the range of 0 to 0.2 to present the trend of performance with respect to $\eta$ more precisely to verify the current selection. You can also add CIFAR-100 and create a curve graph to show the same trend across different datasets.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes Tensor Network Purification (TNP), a defense mechanism designed to reconstruct clean examples from their adversarial perturbations. TNP works by reconstructing progressively downsampled versions of the perturbed input, via a tensor network (TN) decomposition in conjunction with a novel objective that suppresses the reinstatement of the perturbations. The intuition behind this approach is that downsampling with average pooling smooths the perturbative noise towards a normal distribution, which can be effectively handled by the TN decomposition. Unlike previous methods, TNP does not depend on a pre-trained model or a specific dataset, and works as an input pre-process step for any classifier, which does not need to be retrained. The authors perform extensive experimentation, showcasing the effectiveness of the proposed mechanism with respect to various threats and attack types on CIFAR-10, CIFAR-100 and ImageNet.

### Strengths
1.	The experiments and ablation study performed by the authors are very extensive and yield especially good results.
2.	The proposed mechanism is not susceptible to some of the pitfalls of previously proposed adversarial training and adversarial purification methodologies.
3.	The authors provide a clear discussion of TNP’s limitations, which strengthens the paper's contribution and provides clear avenues for future work.

### Weaknesses
1.	A major problem with the current manuscript is that it is somewhat poorly written. First of all, there are various grammatical and syntactical mistakes, as well as typos. For example, see p.1 l.52 “which also implies that allowing for robust defense without retraining the classifier”, p.4 l.211 “we define the linear function ... acts on the image level”, and the title of Section 3 “Backgrounds”.
2.	Even though loosely discussed during the introduction, the general setting and an explicit definition of the problem TNP aims to address are not apparent until halfway through Section 4. As a result, the reader is forced to discover both the method and the problem it is aimed at concurrently. It would have been much smoother had these been formalized before Section 4.
3.	The discussion in Subsection 4.1 is either problematic or calls for additional explanation. In particular, an important assumption of the Central Limit Theorem is that the random variables are independent and identically distributed. The case is somewhat similar in the conditional setting that is referenced. However, since a specific setting is not formalized, it is unclear why this assumption is met. In the same spirit, in Appendix A, average pooling is contrasted with stride sampling so as to support its choice. This is not sufficient support. First, demonstrating the superiority of the method of choice versus a single other alternative does not constitute a principled justification. Second, because both of these are not properly formalized, the results are difficult to actually validate.

### Questions
1.	Can you elaborate on the choice of tensor network decompositions instead of a different kind of model, like a small neural network fitted on a single data point, for example? Have you experimented in this direction? Why do tensor networks deal with perturbative noise that follows the normal distribution?
2.	Can you further explain the reasoning of Subsection 4.1? What assumptions are being made on the distribution of the noise (see the second point in the Weaknesses section)? For example, is it only on the pixel level? Moreover, another point to be made here is that the CLT suggests that an appropriate sequence of random variables converges to a normal distribution. Are the downsampling steps enough to justify invoking the CLT?

### Soundness
2

### Presentation
1

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces TNP, a training-free adversarial defense framework. It addresses the known failure of classical TNs on non-Gaussian adversarial perturbations by hypothesizing that average-pooling downsampling causes these perturbations to approximate a normal distribution. The method proposes a coarse-to-fine optimization pipeline guided by a new adversarial objective, which uses the coarser, cleaner reconstruction as a "surrogate prior" to guide the finer-scale reconstruction. The primary claimed contribution is a data-agnostic purifier that achieves generalization across unseen datasets and attacks compared to existing generative-model-based AP methods.

### Strengths
1.  The paper proposes a training-free framework built on tensor network decomposition. The articulated motivation connects average-pooling downsampling to the Central Limit Theorem as a way to handle non-Gaussian adversarial perturbations.
2.  The method is evaluated in a cross-dataset setting (training on CIFAR-10, testing on CIFAR-100).

### Weaknesses
1.  The paper's evaluation against adaptive attacks is critically flawed. It relies only on BPDA, a known weak attack proxy. A full EOT evaluation is absent, with "memory explosion issues" cited as the reason. This justification is insufficient and raises concerns about gradient obfuscation.
2.  The method's practical utility is low due to poor efficiency. The reported inference time on CIFAR-10 is 2.45s per image, which is noted as slower than diffusion-based baselines and presents a significant barrier to use.
3.  The paper's SOTA claims are not supported by its own data. On the standard CIFAR-10 $l_\infty$ and $l_2$ benchmarks, the method's performance is demonstrably below or on-par with existing works.

### Questions
1.  Regarding the adaptive attack evaluation, could the authors elaborate on the technical barriers (e.g., "memory explosion") that prevented a full EOT evaluation? 
2.  The paper reports a 2.45s inference time on CIFAR-10 (Table 10), while the robust accuracy on this dataset appears on-par with prior work. Could the author discuss more about this cost-performance trade-off?

### Soundness
3

### Presentation
2

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
This paper addresses the limited generalization of adversarial purification defenses by proposing a novel optimization-based defense technique Tensor Network Purification(TNP) method, building upon tensor-based defense strategies. The approach significantly enhances the adversarial robustness of models.

### Strengths
This work presents a novel tensor network-based adversarial purification method, described in substantial detail, which significantly improves the computational efficiency of traditional TN approaches in AP tasks. The experimental analysis is comprehensive

### Weaknesses
1. The defense methods compared in Tables 1–4 vary significantly and lack consistency, which hinders the ability to draw unified conclusions. In particular, it is difficult to assess the specific impact of PTR on SA performance based on the presented comparisons.
2. In Section 4.1, it is mentioned that 512 images were randomly selected for testing. Could author clarify the rationale behind this specific sample size? Additionally, it would be helpful to provide further details regarding the selection process and whether all experiments were conducted exclusively on this subset. The limited sample size raises concerns about potential randomness in the evaluation.
3. Based on the experimental results, PTR appears to suffer from significant overfitting to the distribution of adversarial examples. This is evidenced by a notable improvement in both quantitative and qualitative performance after reconstruction, coupled with a clear decline in the SA metric.
4. Since the proposed defense method operates through a coarse-to-fine iterative process, an analysis of its computational efficiency is necessary.

### Questions
See weakness.

### Soundness
3

### Presentation
3

### Contribution
3
