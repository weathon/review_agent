# Ambient Dataloops: Generative Models for Dataset Refinement

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 2, 6

## Abstract
We propose Ambient Dataloops, an iterative framework for refining datasets that makes it easier for diffusion models to learn the underlying data distribution. Modern datasets contain samples of highly varying quality, and training directly on such heterogeneous data often yields suboptimal models. We propose a dataset-model co-evolution process; at each iteration of our method, the dataset becomes progressively higher quality, and the model improves accordingly. To avoid destructive self-consuming loops, at each generation, we treat the synthetically improved samples as noisy, but at a slightly lower noisy level than the previous iteration, and we use Ambient Diffusion techniques for learning under corruption. Empirically, Ambient Dataloops achieve state-of-the-art performance in unconditional and text-conditional image generation and de novo protein design. We further provide a theoretical justification for the proposed framework that captures the benefits of the data looping procedure.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Since in this process the dataset for estimating the scores at time $t$ for small $t$ is very small, the authors propose an iterative refinement procedure of the dataset. In each step, they use the clean samples and some samples generated from the diffusion model of the previous step. The authors give some intuition about the algorithm and show several examples.

### Strengths
Datasets with partially noisy/degraded elements are a common problem in machine learning and adjusting diffusion models to this case appears to be an important and timely problem. Also the algorithm seems to be tractable and there are plenty of examples. The results are plausible and seem to improve previous methods.

### Weaknesses
- The method starts with a dataset of samples $(x_i,t_i)$, where $x_i$ belongs to the distribution $p_{t_i}$. The paper remains unclear about how to construct this dataset in practice (I noticed that it refers to some literature for this, but for the numerics it should be stated what is done here). 

- The paper often emphasizes that the precise degradation process can be unknown. However, I had the impression that it is required to know which samples are effected by the degradation and how much they are degraded. Correct me if I am wrong, but the paper is not really clear at this point (see also the first question). If this is true, then this has to be clearly addressed in the limitations.

- I am a bit sceptical about the argument that the contractive nature of the OU process really helps. In particular, the reverse of the OU process is an extremely ill-posed problem such that errors in the estimation of the scores will blow up as long as they are not corrected by later score estimates...

- A crucial part of the paper is the approximation that the noisy distribution of the corrupted data almost coincides with the noisy distribution of clean data. The authors should investigate this approximation more in detail. If there is no strong theoretical argument, they could (for instance) investigate numerically in a tractable setting how close this approximation is.

- The paper is not reproducible in the current state. There is no attached code, no detailed description of the implementation and no reproducibility statement. Given that the paper mainly uses empirical arguments of the form "it works better", I consider reproducibility as essential. Also basic information about the required computational resources and the training time is missing.

- Literature: The principle of maintaining a dataset and iteratively refine it is often used in the sampling/fine-tuning literature, see e.g., 
https://arxiv.org/abs/2502.04468
https://arxiv.org/abs/2402.06121
https://arxiv.org/abs/2408.16249
The authors should discuss these works.

- The paper is not really self-contained. Important parts are "outsourced" to the papers of Daras et al. and reading the submission is not straightforward without reading the references. For instance, the sentence "We study exactly the same problem as Daras et al. (2025c;b;a; 2024)" is not sufficient to define the problem setup. Also, the paper does not really make clear what is already there in those references and what is now new.

### Questions
- Does the construction of the dataset require to know which images are corrupted and which are not? And do we need to know "how much" they are corrupted? If yes, this seems to be very restrictive to me. How can this be determined if just a dataset is given and we don't know which images are clean and which ones are corrupted how much (which is the most common case in practice)?

- In their numerics, the authors consider comparibly mild degradations (Gaussian blur, jpeg artifacts). I wondered a bit what kind of degradations can be handeled with this method. Are there some which don't work (e.g. what happens with more severe degradations like salt pepper noise/motion blur/contrast changes)? What does a degradation have to fulfill to be tractable?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces an iterative framework for refining datasets by leveraging generative diffusion models. The core idea is to use a model trained on a noisy dataset to progressively improve the quality of that same dataset. Specifically, the dataset undergoes iterative evolution through diffusion-based techniques to produce increasingly refined versions. This approach demonstrates improvements in the downstream performance of models trained on the refined data.

### Strengths
1. Paper leverages generative models to refine its own training data.
2. The technicality of method is reasonable and well motivated.

### Weaknesses
1. The root of method is mainly based on AmbientDiffusion. Need to justify on the difference.
2. CIFAR experiment is too small and not meaningful (see Table 1) since the gain is quite margnial and the resolution is very low (at 32x32). Need more large-scale experiment on ImageNet.
3. "Our framework trains to extract as much utility as possible from a given training set; if there is more data available, it is always better to perform training updates on it as fresh samples reveal more about the underlying distribution" has no support sentence on how the method is more efficient than restoring the whole dataset.

Paper exposition: 
- Table 5 should be moved to main paper since it is an important evidence.

### Questions
Q1. What is optimal stopping condition (e.g. L)? 

Q2. How to measure the gain of data quality improvement?

Q3. What are eligible points $(x_{\bar{t_1}}, \bar{t_1}), \cdots, (x_{\bar{t_1}}, \bar{t_B})$? How to sample it?

Q4. Have the authors considered a setting that a subset of denoised dataset is being used compared to a whole? Since this allows to show that quality is over quanity and highlights the essense of data quality.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose “Ambient Data Loops”, a method for training diffusion models on a mix of real and noisy data that works by attempting to iteratively denoise the noisy samples. Each step of the loop consists of two components: (1) finetuning on the previous iteration’s data (2) partially denoising the previous iteration’s data. The method builds on prior ambient diffusion works and assumes access to accurate estimates of times $t_i$ for each noisy sample $x_i$ such that training by score matching at $t \in [t_i, T]$ is helpful for $x_i$. In each ambient data loop, the time $t_i$ is halved by using the current model to denoise to time $t_i / 2$ (potentially using multiple samples). 

The algorithm is very roughly theoretically justified without proofs or clearly stated assumptions. In the case that the diffusion model is able to reduce the total variation (TV) distance between the true data distribution and the push-forward of the noisy distribution under denoising, the authors claim the looping procedure will be beneficial. The paper would be stronger if this reduction in TV distance could be theoretically justified.

In Section 5, the authors compare their method to training models that either treat the noisy data as clean (denoted 'No Filtering' in tables) or filter out the noisy data entirely ('Filtering' in tables). Additionally, they compare to ambient diffusion without looping. Results show improvements over No Filtering and Filtering in for both Ambient Diffusion and one Ambient Data Loop. Since there are no confidence intervals and the difference in metrics is small, there is limited evidence that one data loop approves over Ambient Diffusion.

### Strengths
- There is some empirical evidence that one loop is helpful.
- There are a good number of experiments.

### Weaknesses
- Several important experimental details are missing (training time, training hyperparameters, diffusion sampling).
- No confidence intervals or statistical tests are presented in the results tables, so no conclusions about model performance can be drawn.
- Metrics are also under-described. For example, how many samples were used to compute FIDs? These details should be included in the appendix.
- The empirical benefits and contributions are mostly due to ambient diffusion–the looping does not seem to add much and is not theoretically justified due to relying on strong assumptions.
- Section 4 on the theoretical framework is very underdeveloped. Lemma 1 is not a lemma since there is no proof. Maybe it is trivial due to prior results, but this should be justified. Additionally, saying “under appropriate assumptions” without listing those assumptions and proving they are sufficient in the appendix is misleading.

### Questions
- It’s unclear where the protein metrics came from for baseline models. Did you compute them yourself, or take them from the relevant papers?
- Do the theoretical results only apply to a single Ambient Loop?

### Soundness
2

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
3

### Summary
This paper proposes Ambient Dataloops, a training strategy to improve diffusion model training in scenarios where datasets are limited but computational resources are available. The method identifies low-quality or out-of-distribution samples in the dataset, corrupts them with parameterized noise levels, and reintroduces them into the training process with estimated noise strengths. This aims to expand the effective training distribution while preserving controllability. Experiments on controlled CIFAR setups, text-to-image generation, and protein design tasks, supported by ablations, demonstrate the effectiveness of Ambient Dataloops.

### Strengths
- The approach is intuitive and practically motivated, addressing a real challenge in diffusion training when high-quality data is scarce.
- The paper provides theoretical justification explaining why introducing controlled corruption can improve model robustness and training dynamics.
- The empirical evaluation is broad and well-structured, spanning three directions and including ablation studies to demonstrate the contribution of each component.

### Weaknesses
- The real-world evaluation in Section 5.2 is limited to MicroDiffusion. It remains unclear how Ambient Dataloops scale to larger, widely used diffusion models (+ finetuning). For practical adoption, evidence on larger models would be important.

### Questions
- The controlled CIFAR experiments primarily focus on corruptions of in-domain data. How would explicitly OoD datasets (e.g., SVHN added to CIFAR training) affect the model when used in Ambient Dataloops?
    - Would these samples meaningfully expand the effective data manifold, or would they introduce harmful drift, both in baselines and Ambient Dataloops?
- Ambient Dataloops increase training cost by requiring iterative refinement and reintroduction of samples. How do baseline methods behave when given proportionally increased training compute?
    - Are baselines already near saturation, or could additional training diminish the performance gap?

### Soundness
3

### Presentation
2

### Contribution
3
