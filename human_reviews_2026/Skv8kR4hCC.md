# Causal3D: a comprehensive benchmark for causal learning from visual data

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
True intelligence hinges on the ability to uncover and leverage hidden causal relations. Despite significant progress in AI and computer vision (CV), there remains a lack of benchmarks for assessing models’ abilities to infer latent causality from complex visual data. In this paper, we introduce CAUSAL3D, a novel and comprehensive benchmark that integrates structured data (tables) with corresponding visual representations (images) to evaluate causal reasoning. Designed within a systematic framework, Causal3D comprises 19 3D-scene datasets capturing diverse causal relations, views, and backgrounds, enabling evaluations across scenes of varying complexity. We assess multiple state-of-the-art methods, including classical causal discovery, causal representation learning, and large/vision-language models (LLMs/VLMs). Our experiments show that as causal structures grow more complex without prior knowledge, performance declines significantly, highlighting the challenges even advanced methods face in complex causal scenarios. Causal3D serves as a vital resource for advancing causal reasoning in CV and fostering trustworthy AI in critical domains.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Paper introduces Causal3D, a new benchmark that serves to evaluate causal reasoning from observational visual data. Benchmark includes 19 realistic 3D scene datasets comprising of physically consistent "real-world" scenes and hypothetical scenes. 
Work aims to bridge traditional causal discovery and computer vision by aligning tabular data with visual representations. The dataset combines three main causal tasks, namely Causal discovery from tabular data, Causal representation learning from images and Causal discovery & intervention from few images.
Authors also provide comprehensive evaluation across traditional causal discovery algorithms, causal representation learning models, and modern Vision-Language Models (VLMs) such as ChatGPT, Gemini, and Claude across the various tasks.

### Strengths
CAUSAL3D is the first dataset combining explicit causal graphs and realistic 3D visual representations, filling a key gap in causal learning evaluation.

Work has a clear motivation in aiming to bridge the gap, with comprehensive evaluations for each task across several methods. 
The authors benchmark a diverse range of models, from classical causal discovery to VAE-based causal representation learning and commercial VLMs.

Inclusion of multi-view images and analysis on the impact gives a novel perspective into causal discovery from from visual representations.

### Weaknesses
1. Despite claims of physical consistency, the benchmark remains entirely simulation-based. Simply using realistic backgrounds does not equate to real-world data capture, and this should be stated more cautiously. 
2. Claims of different fundamental physical principles (line 208 -211) with differing number of variables and unique causal structures. Would like to have seen more statistics and details on the dataset to support the claims in "different dimensions, including the number of variables (ranging from 2 to 5), multiple causal structures, different (linear/nonlinear) types of causal relations, and various camera views and backgrounds in 3D scenes"
3. Under section 4.3 of Causal Representation Learning discussion is only on 2 qualitative results that are presented for evaluation. It would have been more beneficial if analysis were based on more samples across the entire dataset.
4. Again under Views and Backgrounds, the analysis would have been better supported with qualitative results. i.e multi-view vs single-across different scenes, and performance virtual and realistic backgrounds.

Minor comments\
Generally in section 4, it would have been clearer to have represented the numerical values of the results tabular.

### Questions
My main concern lies with the claims and analysis of the performance of current methods on the benchmark to support claims of the benefits of the dataset. 

The work as a whole is well motivated and could serve as a foundational benchmark for causal reasoning research within computer vision. However, the current empirical analysis does not sufficiently substantiate the paper’s claims.
The causal representation learning section is particularly weak, relying almost entirely on qualitative samples without quantitative validation or comprehensive evaluation across the dataset. Furthermore, the lack of comprehensive details and analysis on their own dataset makes it difficult to substantiate their claims as stated in point 2 of weakness.

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
4

### Summary
This paper introduces Causal3D, a comprehensive benchmark designed to evaluate causal reasoning capabilities from visual data. The benchmark is built on 19 distinct 3D-rendered scenes that combine realistic imagery with explicit, underlying causal structures (DAGs and structural equations). The authors conduct extensive evaluations across three causal tasks—causal discovery from tabular data, causal representation learning from images, and causal discovery from few images using VLMs—demonstrating the benchmark's utility and revealing significant challenges for current methods, especially as causal complexity increases.

### Strengths
1)	The introduction of Causal3D fills a significant void. Existing datasets are either purely tabular, lack explicit causal graphs, or are too simplistic. Causal3D's integration of realistic 3D visuals with rigorous causal ground truth is a major contribution that will facilitate research at the intersection of computer vision and causal inference.

2)	The benchmark is meticulously designed for progressive evaluation. The inclusion of multiple views and backgrounds (real/virtual) allows for nuanced analysis of model robustness to spurious correlations, a critical aspect of causal learning.

### Weaknesses
1)   The benchmark, in its current form, is focused solely on observational data. A crucial aspect of causal inference is interventional and counterfactual reasoning. The authors acknowledge this in the limitations, but its absence currently restricts the benchmark's scope. Incorporating interventional data (e.g., by allowing users to manipulate variables in the 3D engine) would make it even more powerful.

2)	The evaluation of VLMs on the causal discovery task, while interesting, could be clarified. The prompt asks the model to output an adjacency matrix, which is a highly structured and non-standard output for these models. It would be beneficial to discuss potential parsing issues and how the models' raw text outputs were consistently mapped to a matrix format. Furthermore, a comparison against a simple baseline (e.g., a model that always predicts no edges) would help contextualize the absolute performance scores, which are often quite low.

3)	The paper would benefit from a more explicit justification for why the specific 8 real-world and 11 hypothetical scenes were chosen. A discussion on how they cover a representative space of causal structures (e.g., chains, forks, colliders) would strengthen the claim of comprehensiveness.

4） For the causal representation learning section, the choice of evaluation scenes for each model (e.g., why CDG-VAE was evaluated on reflection and pendulum but not spring) is mentioned but could be more clearly motivated upfront.


5） Figures 3, 5, and 6 are referenced in the text but were not included in the provided content. Their absence makes it difficult to fully assess the experimental findings. The review assumes these figures effectively support the claims.

6）The analysis of multi-view vs. single-view performance is insightful but could be taken a step further. A hypothesis or deeper discussion on why multi-view helps in complex scenes but hurts in simple ones would be valuable (e.g., is it due to overfitting, or the introduction of conflicting visual cues?).

### Questions
See the weaknesses.

### Soundness
3

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
3

### Summary
This paper introduces Causal3D, a new benchmark for causal learning from visual data. It includes 19 realistic and hypothetical 3D scenes combining images, tabular data, and causal graphs. The benchmark aims to test models on causal discovery, representation learning, and intervention. The dataset is well-structured and systematically designed. Experiments with classical methods, causal VAEs, and recent methods highlight current limitations in visual causal reasoning.

### Strengths
- The dataset is comprehensive covering both physical and hypothetical causal structures.
- Several types of experiments with various methods (tabular, image, multimodal).
- The work aim to provide insides on the limitations of current causal and vision-language models.

### Weaknesses
- The benchmark still deals with simple, static scenes with few objects and low-dimensional latent variables. It lacks dynamic data or temporal causality, which limits real-world relevance.
- There is no reference or comparison to “Weakly Supervised Causal Representation Learning” or to the Causal Circuit Dataset, which are relevant benchmarks.
- The evaluation scope focuses mainly on 2D rendered scenes and does not test robustness under more realistic 3D dynamics or noise.
- It is unclear how to extend the benchmark to higher-dimensional or real-world videos.

### Questions
- Do you plan to extend Causal3D to dynamic or temporal datasets?
- Could you discuss why prior benchmarks like Causal Circuit (Weakly Supervised Causal Representation Learning) were not referenced or compared?

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
This paper presents CAUSAL3D, a novel benchmark designed to evaluate causal learning from complex visual data. CAUSAL3D includes multiple physically consistent and hypothetical scenes, providing data in both tabular and image modalities. Furthermore, it supports intervention settings. The authors conduct a comprehensive evaluation, benchmarking traditional causal discovery algorithms and LLM-based methods. The evaluation involves several scenarios including Causal Discovery from Tabular Data, Causal Representation Learning, and Causal Discovery from Few Images, providing critical insights into the limitations of current SOTA models.

### Strengths
- The dataset covers a diverse range of scenarios, including both tabular and image-based data, physically consistent scenes and hypothetical scenes.
- The evaluation is comprehensive, benchmarking both traditional causal discovery methods and modern LLM-based approaches across multiple tasks.
- The paper is well-written and clear.

### Weaknesses
- The paper fails to discuss the  Synthesis and Reality gap. It is unclear whether this synthetic dataset can accurately reflect causal relationships found in real-world data or if it can benefit research on real-world problems.
- The discussion on the construction of the causal data lacks detail regarding the noise term. It is not specified how the noise term is defined or how different noise levels might impact the evaluation results.

### Questions
- Could the authors discuss the dataset's generalization to real-world cases? For example, can models trained on this dataset using supervised learning successfully perform causal inference on real data?
- Could the authors provide a more detailed discussion of the noise term? For instance, it would be helpful to see evaluation results under different noise strengths.

### Soundness
3

### Presentation
3

### Contribution
3
