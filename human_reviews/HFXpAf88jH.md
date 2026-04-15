# Beyond Implicit Bias: The Insignificance of SGD Noise in Online Learning

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 6, 5

## Abstract
The success of SGD in deep learning has been ascribed by prior works to the *implicit bias* induced by high learning rate or small batch size ("SGD noise"). While prior works  that focused on *offline learning* (i.e., multiple-epoch training), we study the impact of SGD noise on *online* (i.e., single epoch) learning. Through an extensive empirical analysis of image and language data, we demonstrate that large learning rate and small batch size do *not* confer any implicit bias advantages in online learning. In contrast to offline learning, the benefits of SGD noise in online learning are strictly computational, facilitating larger or more cost-effective gradient steps. 
This suggests that SGD in the online regime can be construed as taking noisy steps along the "golden path" of the noiseless *gradient flow* algorithm. We study this hypothesis and provide supporting evidence in function space by conducting experiments that reduce SGD noise during training and by measuring the pointwise functional distance between models trained with varying SGD noise levels, but at equivalent loss values. Our findings challenge the prevailing understanding of SGD and offer novel insights into its role in online learning.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
1: You are unable to assess this paper and have alerted the ACs to seek an opinion from different reviewers.

### Summary
This paper investigates the influence of SGD noises, specifically batch size and learning rate, on implicit bias within an online learning context. Through comprehensive experiments, the authors demonstrate that unlike in offline settings, SGD noise does not confer any additional advantages in online learning.

Furthermore, the authors introduce and explore the "golden path hypothesis" in relation to online learning. Empirical analysis suggests that for real-world data utilizing deep neural networks, a "noiseless" or "golden" path trajectory may be present, implying that SGD could potentially mimic the trajectory of gradient flow algorithms.

### Strengths
The problem studied in this paper is important as the LLMs might adopt the examined online method to update their parameters. This paper performs extensive experiments to support their emperical findings.

### Weaknesses
The online learning setting investigated lacks a rigorous and detailed formulation.  See more details in Questions.

### Questions
1. The online learning protocol discussed in this paper is not entirely clear to me. Could the authors provide with a more detailed formulation of the online learning procedure using SGD? In the online learning contexts I'm familiar with, such as Prediction with Experts' Advice, regret is typically employed as a performance measure. Could the authors clarify how the algorithm's loss is assessed in the online learning setting under consideration?

2. I'm curious about the relationship between the convergence rate and the choice of adaptive learning rate. Is the observed behavior consistent when using optimizers like Adam?

3. How does this research account for or negate the effects of the neural network's architecture?

4. I'm interested in understanding the design of the experiments. Given that in the real-world online learning setting, achieving comparable performance can be more challenging without full access to the dataset, yet it offers efficiency advantages. Were there particular measures or modifications incorporated to guarantee an fair comparison with the offline setting?

While empirical studies involving SGD algorithms fall outside my primary domain of expertise, I am open to further discussions on the topic.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper studies the potential implicit bias effect of SGD noise in online learning. The authors observe from experiments that in online learning settings. SGD noise does not bring any implicit bias and it is "just noise".  Next, based on experiments, the authors also proposed the "golden path" hypothesis, which states that SGD with different noise levels follows the same trajectory (which they call "golden path") in function space in online learning setting. The authors also perform experiments to support their hypothesis.

### Strengths
1. The main result in this paper that SGD follows the same path in online learning settings is an interesting finding in my opinion.

2. The experiments support the main claims well, and the claims made by the paper are stated clearly in general.

### Weaknesses
1. I would like to understand more about the scope of the main results:

- The experiments are performed on Resnet18, ConvNext-T, and GPT-2 small, which are relatively large models. I'm wondering if the main hypothesis of this paper also holds for smaller models, or if this phenomenon might be due to the overparameterization of the models?

- The study of this paper focuses on SGD noise, i.e. the noise comes from not using full-batch. I'm wondering if the main hypothesis also holds for manually added noise (e.g. noisy gradient descent like Langevin dynamics) ?

- A minor point: In the paper, your main findings and hypothesis are made for SGD, while in your experiments, the optimizers used are SGD with momentum (for ResNet18), AdamW (for ConvNext-T), and "default optimizer in Mosiac ML" . So the main hypothesis is not only for SGD but also for difference optimizers?

- As you mentioned in the discussion at the bottom of page 2, the "golden path" is the noiseless gradient flow. I'm wondering if you could compare the trajectory of SGD to the actual gradient flow (i.e. GD with full batch, and very small step size) ?

2. The experiments on reducing the step size are not very clear to me, since the step size also affects the sharpness of the solution SGD can find (as you the "edge of stability" phenomenon you mentioned in Appendix D). So it seems to me that the decrease of loss after decreasing the step size may be due to the fact that the dynamic is around a local minimum of certain sharpness, and a smaller step size allows it to go into this local minimum, rather than a better approximation of the "golden path" due to smaller SGD noise. Similar arguments could also made for the experiments on increasing step size.

### Questions
Please refer to the strengths and weaknesses part.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper carries out a series of experiments to compare offline learning (multiple-epoch training) with online (single epoch) learning. The experiments are based on commonly used image and language data and the focus of the experiments is the role played by "SGD noise". High SGD noise refers to high learning rate or small batch size. Unlike the offline learning, the benefits of SGD noise are not observed in the experiments for online learning. It is conjectured that SGD in the online learning case can be interpreted as noisy learning along the "golden path" of the noiseless gradient flow algorithm.

### Strengths
1. It is very interesting that SGD noise plays a different role between single and multiple epoch regimes.

2. Figures are well-presented and convey succinct summary of experimental results.

3. The expressions "Fork in the Road" and "Golden Path" are eye-catching terms that create instant curiosity.

### Weaknesses
1. The paper is mostly well written; however, the details behind the experimental results are somewhat sparse, including the appendix. Some further clarifications would strengthen the paper substantially. For example, on page 5, it is stated that "To imitate the online regime with ImageNet, we only train for 10 epochs with data augmentation." In the abstract, online learning refers to the single epoch regime but on page 5, it seems that this is not the case. Furthermore, Appendix A contains very short explanations for each of experiments. It is hard to understand exactly what was done in the experiments given the sparse information provided in the paper.

2. All the claims in the paper are entirely driven by the experiments; there are no theoretical results. It would be more prudent if the author(s) could provide the limitation of the current paper on page 9.

### Questions
1. It is unclear how many epochs are considered in multiple-epoch training across different experiments. For example, in Figure 1, the top and bottom rows, respectively, show the results from offline learning (multiple-epoch training) and those from online learning (single-epoch training). The training steps are on the same scale between the top and and bottom rows. In the case of top rows, there is no indication of how many epochs are considered. It would be helpful to provide further details. 

2. Related to the previous point, is it OK to interpret the X axis the same way between the top and bottom figures in Figure 1? For example,  would it be possible that the patterns observed in offline learning can appear if the number of training steps in online learning is much larger, say, 10 or 100 time larger than 4000? The early paths observed in offline learning are quite similar to those observed in online learning. 

3. In addition, what are details of multiple-epoch training? Is multiple-epoch training conducted via random shuffling of the datapoints after each epoch or a simple random sampling of data points with replacement at each step (or something different)? Again it would be helpful to understand the exact nature of multiple-epoch training.

4. The provided supplementary material does not include replication files. Given that the current paper is experimental, it would be useful if all replication files are provided on public domain.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair
