# Exploring weightless neural networks: From logic gates to convolutional lookup tables

- Decision: Reject
- Scores: 6, 4, 2

## Abstract
Miniaturizing Machine Learning (ML) models to operate accurately in resource-constrained environments may improve the intelligence of everyday objects. Applications abound across the Internet of Things (IoT), for personal medical devices, and for consumer electronics such as smartphones and augmented reality glasses. Differentiable Weightless Neural Networks (DWNN), such as differentiable Logic Gate Networks (LGN) and Look Up Table networks (LTN), represent a class of models which accelerate ML inference by orders of magnitude while retaining high predictive accuracy. While small-scale LGNs and LTNs already expedite model inference and reduce resource usage, performance and robustness benchmarks are not well reported which hinders the development of large architectures suitable for real-world applications. This paper fills this gap by comparing LGNs, LTNs, Multi-Layer Perceptrons (MLPs), and their convolutional counterparts on the basis of test accuracy, training time, and robustness to noise across key model and training variations. We introduce the Look-Up Table Convolutional Network (LTCNN), which reduces training time by 2–4X compared to prior logic-based convolutions. By benchmarking over 4,000 models, we identify critical scaling limitations: unlike standard CNNs, increasing DWNN parameter counts tends to exacerbate brittleness to environmental noise rather than improving generalization. Furthermore, while employing learnable interconnects improves accuracy by 6–20\%, it incurs a 3X computational penalty. These results quantify the distinct trade-offs of weightless architectures, highlighting the need for training strategies to scale DWNNs for real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper conducts an extensive empirical study on Weightless Neural Networks (WNNs), particularly the Logic Gate Networks (LGNs) and Look-Up-Table Networks (LTNs), exploring their scalability, robustness, and training efficiency relative to standard MLPs and CNNs. The authors introduce a convolutional variant, the LTCNN, designed to mimic CNN kernels via sliding-window logic, and evaluate it against existing LGCNNs.

Three systematic studies are presented:

1. Model Scaling: Comparing training time, accuracy, and noise robustness across model sizes.
2. Bit-Depth Variation: Assessing how quantization granularity (1-, 2-, 4-bit) affects performance.
3. Learnable Mappings: Investigating the impact of learnable interconnects between logic layers.

Results across MNIST, Fashion-MNIST, and CIFAR-10 show that WNNs achieve comparable accuracy to traditional DNNs on simple datasets but require larger parameter counts and training time. LGNs display superior robustness to salt-and-pepper noise, while LTNs generally train faster. However, scaling beyond modest architectures remains challenging due to combinatorial training complexity and limited receptive fields.

### Strengths
*Unprecedented experimental scale:* Over 3000 model variations evaluated across architectures, datasets, and encoding schemes — the largest comparative WNN study to date.

*Methodological clarity:* Parameter search, optimization settings, and training details are exhaustively documented, ensuring reproducibility.

*Architectural innovation:* Introduction of LTCNNs extends WNN applicability to spatially structured data.

*Balanced analysis:* Includes multiple performance metrics — accuracy, training time, and robustness — not just raw accuracy.

*Hardware relevance:* Considers inference efficiency for FPGA deployment, highlighting edge-device applicability.

### Weaknesses
*Limited conceptual novelty:* Despite broad experimentation, the contribution is primarily empirical — no new training paradigm or theoretical framework is proposed.

*Underdeveloped scalability discussion:* The paper identifies training inefficiency but doesn’t analyze why gradient-based optimization underperforms with discrete structures.

*Missing SOTA comparisons:* Lacks benchmarks against Binary Neural Networks (BNNs) or quantized models (e.g., XNOR-Net, DoReFa-Net), which target similar hardware-efficient goals.

*Overemphasis on small datasets:* Evaluation restricted to MNIST-family and CIFAR-10 — too elementary for claims of “real-world scalability.”

*Ambiguous bit-depth insights:* The bit-depth study’s findings (“depends more on dataset than model”) feel descriptive rather than explanatory.

*Unclear path forward:* Future work is listed but not tied to the limitations uncovered, weakening the narrative closure.

### Questions
*Detailed Analyses:*

This paper stands at the crossroads of symbolic determinism and differentiable learning. It is not just a technical benchmark but a philosophical probe into how much “logic” can live inside a modern neural framework.

The study’s brilliance lies in revealing that weightlessness is not simplification — it’s structure exposed. The very mechanisms that make LGNs interpretable — fixed binary operators and explicit mappings — also constrain their ability to scale. This is the paradox of discrete differentiability: transparency breeds rigidity.

Yet, the work’s contribution is not diminished by its empirical focus. It charts the limits of current WNNs while providing an honest, data-driven narrative of their trade-offs. It implicitly calls for hybridization — integrating logic-based regularization or attention-like symbolic layers into conventional deep nets.

In short, the paper answers a deeper question: where do Boolean ideals meet the entropy of gradient descent? And in that meeting, it maps the next horizon of neurosymbolic research.
While not theoretically groundbreaking, this paper’s scale, rigor, and insight make it a valuable empirical cornerstone for the neuro-symbolic community. Its clarity and reproducibility elevate it beyond a routine benchmarking effort. However, it would benefit from stronger engagement with recent SOTA baselines and a more principled discussion of why weightless architectures hit their current limits.

I expect the authors to defend or rebut the points in the weakness section during the rebuttal phase.

### Soundness
3

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
This paper presents a comprehensive investigation of Weightless Neural Networks (WNNs), specifically Logic Gate Networks (LGNs) and Look-Up Table Networks (LTNs) and compares them to conventional neural models (MLPs and CNNs). It explores a very wide range of model configurations (>3000), analyzing their impact on training time, accuracy, and noise robustness using 3 image-based datasets (MNIST, Fashion-MNIST, CIFAR-10). Impact of learnable mapping (trainable inter-connects) and bit depth of inputs encoding is also studied. As part of their evaluation, the authors also introduce a novel LTN architecture (LTCNN), by applying to LTN the sliding-window mechanism characteristic of LGCNN, an LGN variant.
Results show that, at the range of model sizes investigated, LTNs and LGNs achieve comparable accuracies and noise robustness to their MLP and CNN counterpart, although requiring longer training times. The optimal bit depth is primarily dataset-dependent. Learnable mapping can be beneficial for accuracy but at the cost of significantly increased model size and training time.

### Strengths
- The main strength of this paper is the comprehensiveness of the comparative study of LGNs and LTNs, an exploration covering a very wide range of model configurations and test conditions. The results offer a consolidated reference for WNN performance
- The paper is well structured and results are well organized. It is generally easy to follow, although some concepts, such as learnable mapping and sliding-window modification for LGCNN, are taken for granted and not explained for a general audience
- The authors do not overclaim, they offer a balanced discussion of WNNs underperforming/overperforming compared to the counterpart reference models

### Weaknesses
- Limited novelty: the primary novelty lies in 1) the introduction of the LTCNN and 2) an extensive configuration sweep. However, LTCNN is conceptually a direct adaptation of the existing LGCNN. It should be noted that the exact sliding window mechanism the authors introduce in LTCNN is not described in details in this paper, although it is understood to be equivalent to the one used in LGCNN. LTCNN do not appear to offer significant performance gains and are slower to train. On the other hand, the broad hyperparameter exploration is not a source of novelty per-se, and the discussion is primarily observational, with speculative explanations
- In terms of impact, a key bottleneck to WNNs practical applicability is the long training time and this paper confirms this limitation rather than offering a solution or mitigation strategy. Consequently, experiments relies on very small-scale image datasets and small models (up to 1M parameters), severely limiting generalization to real-world or large-scale data
- Other explorations (noise robustness, bit width) show mixed results, in the sense that different trends are observed across models. Although interesting, they suffer from the same lack of generalizability to larger datasets and more challenging tasks
- Learnable mappings improve performance but exacerbates the fundamental limitation of WNN, the long training time, further worsening scalability

### Questions
- Can LTCNN be optimized to improve training time, similarly to kernel optimization implemented for LGCNN in the cited Petersen et al. 2024?
- Can the observed trends be generalized to more complex datasets, at least in some scenarios like noise robustness?

### Soundness
3

### Presentation
3

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
This paper presents an empirical comparison of Weightless Neural Networks (WNNs)—Logic Gate Networks (LGNs) and Look-Up Table networks (LTNs)—against traditional deep neural networks (MLPs and CNNs). The authors train 1040+ model architectures across MNIST, Fashion-MNIST, and CIFAR-10 to evaluate test accuracy, training time, and robustness to noise.

### Strengths
Training 1040 architectures across three datasets with multiple evaluation dimensions (accuracy, training time, robustness) represents a significant experimental effort.

Introduction of convolutional LTN variants fills a gap in the literature and enables fair comparison with LGCNNs.

The paper addresses real engineering questions (training time, robustness, bit depth) relevant to practitioners considering WNN deployment.

Beyond accuracy, the robustness analysis (salt-and-pepper noise, occlusions) and training time measurements provide valuable practical insights.

The paper also tests Fashion-MNIST, which was not done by Petersen et al. (2022;2024), and something that was missing in their evaluation.

### Weaknesses
The statement “In real-world deployments, applying augmentation would likely improve performance” should simply be tested. 

The paper's core motivation is FPGA deployment and inference speed, yet never measures either.
All experiments run on GPUs (NVIDIA L4)
No inference time measurements reported
No hardware resource utilization (LUTs, power consumption)
No comparison to actual FPGA implementations
Some or all of these are critical to draw the real-world conclusions the authors do.

The statement “Note that LGNs and LTNs achieve state-of-the-art performance for MNIST and Fashion-MNIST (i.e. hand written characters and clothing items) while performing worse on CIFAR-10 (i.e. containing structurally complex images of birds, cars, and other classes), allowing these datasets to stress each model’s performance and reveal challenges with training complex model architectures” requires citations. 

No error bars on accuracy measurements despite stochastic training

2-fold validation is unusual—why not standard 80/10/10 or 5-fold cross-validation?
Averaging over "top 5 models" biases results toward best-case scenarios

Some missing related work. A few of these are merely concurrent work, but it makes sense to cite given the overlap.
https://arxiv.org/abs/2508.17512 
https://arxiv.org/abs/2506.07500 (you already mean to cite this. It is the Yousefi & Wattenhofer 2025 citation) 
https://ieeexplore.ieee.org/document/10301592 
https://arxiv.org/abs/2510.03250 
https://arxiv.org/abs/2506.04912 
https://arxiv.org/abs/2509.25933 
https://arxiv.org/abs/2504.00592

You cite “Shakir Yousefi and R Wattenhofer. Deep differentiable logic gate networks: Neuron collapse through a neural architecture search perspective. 2025.” However, this is a project description. Yousefi published their work in the Mind the Gap paper (https://arxiv.org/abs/2506.07500).

Captions for the tables should be above the tables as per the formatting instructions. 

The figures are generally low resolution, and the font is small. Please address this. 

The story of the paper is interesting; however, the writing is rather clunky, and the presentation could be improved. This is in particular the case for sections 3.2 and 4.

The color-coded bars in Tables 2 and 3 are hard to interpret. 

Typos:
Line 50 should have “ML” rather than “Ml.”


While my review is rather negative, the authors can and should address several of these things for the iclr submission, as the paper and reviews will be public. The issues with the citations, missing citations, figures, captions, etc., can be resolved within a day :)

### Questions
What is the training and validation split? Line 187 makes it sound like 50/50, but this seems quite aggressive.

Why did you leave out all data augmentations? I understand omitting some to test generalization to unseen perturbations; however, if you want to determine their real-world applicability, then this should be included. 

Why did you use the quantization method over a temperature encoding as Petersen et al. use?

What is the full distribution of accuracies (not just top-5)?

Can you create a figure for section 3.2, as it is currently not easy to understand?

How many layers do the models have? (e.g. in Table 1).

Do you know why LTCNN’s time per epoch drops a lot for the largest models in Table 1. Both the mean and the std are very low.

Why are your CNN on CIFAR results so poor? It should not be hard to get an accuracy around 80% (https://www.kaggle.com/code/faressayah/cifar-10-images-classification-using-cnns-88) 

You report test accuracies for your DWN that are much lower than in the original DWN paper. Your Table 3 Fashion MNIST test accuracies are around 55% while the DWN paper reports 89% (see Table 1 https://arxiv.org/pdf/2410.11112). Why is this?

### Soundness
2

### Presentation
1

### Contribution
2
