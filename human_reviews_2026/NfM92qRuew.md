# Intermediate Representations are Strong Training-Free AI-Generated Image Detectors

- Avg Score: 3.33
- Decision: Reject
- Scores: 2, 4, 4

## Abstract
The rapid advancement in generative AI models has enabled the creation of photorealistic images. At the same time, there are growing concerns about the potential misuse and dangers of generated content, as well as a pressing need for effective AI-generated image detectors. However, current training-based detection techniques are typically computationally costly and can hardly be generalized to unseen data domains, while training-free methods fall short in detection performance. To bridge this gap, we propose a training-free method employing data embedding sensitivity in intermediate layers to detect AI-generated images. Given a set of real and AI-generated images, our method scans through the detection performance in the composite configuration space of intermediate layer, perturbation type, and severity level to identify the best configuration for detection. We examine the proposed method on two comprehensive benchmarks: GenImage and DF40. Our method exhibits improved performance across different datasets compared to both training-free and training-based state-of-the-art methods. On average, our method outperforms the best training-free/training-based methods on the GenImage benchmark by 16.1%/4.9% and on the DF40 benchmark by 14.5%/8.7% in AUROC score. We release the code at https://anonymous.4open.science/r/Intermediate-Public-D256.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper proposes a training-free AI-generated image detection method, which extends previous training-free methods that utilize the difference between the data embedding sensitivity of real and fake images. The proposed method performs a grid search on the configuration space of the intermediate layer, perturbation type, and severity level to identify the best configuration, based on a subset of the training data. Experimental results on GenImage and DF40 show the superiority of the proposed method.

### Strengths
1. The analyses on the differences across intermediate layers are interesting.
2. The proposed method is clearly and formally presented.

### Weaknesses
1. While the proposed method is claimed to be training-free and to have good generalization performance, it is suspected to be reliant on training data, and the cross-generator generalization is not validated.
  - The selection of the optimal configuration is based on a subset of training data.
  - According to Sec. A.2, the optimal configurations are selected on a **per-generator** basis. In other words, the experiments in this paper assume that the training and test AI-generated images are produced by the same generator. The cross-generator generalization performance is not studied.
  - As suggested by Sec. A.1, the baseline methods are not trained in a per-generator manner. This leads to an unfair comparison with the proposed method.
2. The experiments are conducted solely on GenImage and DF40, where advanced DiT-based image generators such as FLUX and Stable Diffusion 3 are missing. Testing on more challenging and up-to-date datasets such as Chameleon [1] and Community Forensics [2] is important for validating the practical performance of the proposed method.
3. Figure 1(c) is confusing. This 3D mesh plot seems to represent how the "layer" (z-axis) changes along with the severity level and perturbation type, while the color is determined by the detection performance. The "optimal configuration" points to a local minimum. It requires further clarification.
4. The technical contribution is limited as the proposed method only extends previous training-free methods by grid searching on several key factors for an optimal configuration.

[1] A Sanity Check for AI-generated Image Detection. ICLR 2025.  
[2] Community Forensics: Using Thousands of Generators to Train Fake Image Detectors. CVPR 2025.

### Questions
1. Is the proposed method still effective for the cross-generator scenario? For example, when the training data is limited to one or a few generators, while the test data involves images generated from unseen generators? Can you determine a universal optimal configuration for different test data?
2. Is the proposed method robust to different types of input perturbations?

### Soundness
1

### Presentation
2

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
The paper proposes a training-free detector that searches over a configuration space (layer index, perturbation type, severity) and classifies by cosine similarity between an image and its perturbed version using intermediate features from a frozen foundation model (mainly CLIP and DINOv2). Selection of the “best” configuration uses a labeled subset of the training set; inference then applies that single configuration to test images. Claims include sizable AUROC gains on GenImage and DF40 over both training-free (RIGID, MINDER) and several training-based baselines.

### Strengths
1. The proposed framework is a simple and code-free adaptation to different backbones (CLIP/DINOv2). The method and Algorithm 1 are easy to reproduce.

2. Intermediate-layer analysis is informative and justifies looking beyond the last layer.

### Weaknesses
1. The core paradigm, i.e., comparing feature similarity between an image and a perturbed version, is already central to RIGID and MINDER, which the paper also acknowledges. The method mainly expands the search space (more layers, more perturbations/severities). This reads as an incremental extension rather than a new principle.

2. Stage I requires labeled data and computes similarities across up to 1000+ configurations to pick one (30% of test-set size by default). That is a non-trivial, dataset-specific tuning cost and blurs the “training-free” positioning. Moreover, the paper reports generator-specific best configurations (Tables 3–4), which suggests per-domain tuning. In other words, Tables 3–4 show each generator has a different best (layer, perturbation, severity). In practice, the generator is unknown; tuning per generator (or per image) is infeasible. 

3. The paper lacks tests on real capture degradations and unbiased/realistic testbeds such as Chameleon [a] and GenImage-unbiased [b] that specifically probe shortcut/bias failure modes.

[a]A Sanity Check for AI-generated Image Detection
[b] Fake or JPEG? Revealing Common Biases in Generated Image Detection Datasets

4. The paper only focuses on AUROC. While for operational utility and comparison with prior works, please also report Accuracy, Average Precision (AP), which are commonly used in this area.

### Questions
1. In terms of Weakness-1:  Please (1) clarify what is fundamentally new beyond search and \psi (Eq. 2), and (2) provide ablations that show why weighted or searched multi-layer choices outperform a strong single-layer baseline under the same corruption budget, and (3) include a “RIGID-with-search” control (same search over severity/types but using only the final layer) to isolate the contribution of intermediate layers.

2. In terms of Weakness-2: (1) Report latency/compute and wall-clock for Stage I vs. a one-pass inference of training-based baselines (e.g., NPR, AIDE) to substantiate the “training cost” motivation. (2) Evaluate a universal configuration chosen on one source set and frozen across unseen generators/datasets (cross-domain transfer), to approximate real deployment.

3. Weaknesses 3-4, I list the questions in the comments.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a "training-free" method for detecting AI-generated images by applying perturbations, extracting intermediate-layer features, and measuring sensitivity differences between original and perturbed representations to separate real from generated images. The authors evaluate ~1,600 configurations, select the best using limited labeled data, and show on GenImage and DF40 that this approach outperforms existing training-free and some training-based detectors in AUROC.

### Strengths
- The method does not require modifying the backbone; it only relies on extracting intermediate-layer representations from vision foundation models, making it straightforward to implement and reproduce.
- The paper is clearly written and logically structured.
- The experimental results are impressive, in some settings even surpassing carefully trained detectors.

### Weaknesses
- Existing training-free detectors (e.g., RIGID/MINDER) have already exploited the idea that “real images and synthetic images exhibit different robustness under perturbations.” The main difference in this work is that, instead of examining only the final layer or a single perturbation, it treats the tuple (intermediate layer index × perturbation type × perturbation strength) as a large discrete hyperparameter space, and then searches for the optimal combination. This is an incremental improvement in engineering.
- Although the authors repeatedly emphasize that the method is “training-free,” it still requires labeled images (both real and AI-generated) from approximately 30% of the training set. These images are used to evaluate ~1600 configurations and select the one that yields the best AUROC, which is then deployed for inference.
- It remains unclear how the method behaves when more diverse backbones are considered, and whether visual foundation models with ConvNeXt-style architectures would still be applicable.

### Questions
**Typos**: 

- In Equation (2), there is an extra closing parenthesis “)”.
- In the caption of Figure 7, “GenImage benchmark” should be “DF40 benchmark”.
- In Table 3 and Table 4, “JPEG compress” should be changed to “JPEG compression”, to be consistent with the main text.

### Soundness
3

### Presentation
3

### Contribution
2
