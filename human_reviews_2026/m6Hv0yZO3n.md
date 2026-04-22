# Discount Model Search for Quality Diversity Optimization in High-Dimensional Measure Spaces

- Avg Score: 5.50
- Decision: Accept (Oral)
- Scores: 4, 4, 8, 6

## Abstract
Quality diversity (QD) optimization searches for a collection of solutions that optimize an objective while attaining diverse outputs of a user-specified, vector-valued measure function. Contemporary QD algorithms are typically limited to low-dimensional measures because high-dimensional measures are prone to distortion, where many solutions found by the QD algorithm map to similar measures. For example, the state-of-the-art CMA-MAE algorithm guides measure space exploration with a histogram in measure space that records so-called discount values. However, CMA-MAE stagnates in domains with high-dimensional measure spaces because solutions with similar measures fall into the same histogram cell and hence receive the same discount value. To address these limitations, we propose Discount Model Search (DMS), which guides exploration with a model that provides a smooth, continuous representation of discount values. In high-dimensional measure spaces, this model enables DMS to distinguish between solutions with similar measures and thus continue exploration. We show that DMS facilitates new capabilities for QD algorithms by introducing two new domains where the measure space is the high-dimensional space of images, which enables users to specify their desired measures by providing a dataset of images rather than hand-designing the measure function. Results in these domains and on high-dimensional benchmarks show that DMS outperforms CMA-MAE and other existing black-box QD algorithms.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes Discount Model Search, a Quality-Diversity algorithm to address stagnation in high-dimensional measure spaces caused by discrete histograms in state-of-the-art methods like CMA-MAE. DMS uses a smooth neural network-based discount model to distinguish solutions with similar measures, guiding exploration effectively. It also introduces the Quality Diversity with Datasets of Measures setting, where users specify desired measures via datasets instead of hand-designing measure functions. DMS is evaluated on 7 benchmark domains and 3 QDDM domains , outperforming baselines like CMA-MAE, DDS, and MAP-Elites in most cases.

### Strengths
1. By replacing CMA-MAE’s discrete histogram with a continuous, learnable discount model, DMS addresses the limitation of identical discount values for similar measures in high-dimensional spaces.
2. The QDDM framework eliminates the need for tedious manual measure function design by leveraging datasets. This expands QD’s applicability to vision and creative domains.
3. Experiments cover both standard QD benchmarks (testing distortion and scalability) and novel QDDM tasks (testing real-world applicability). Ablation studies rigorously verify key components (e.g., empty points prevent arbitrary discount values, archive learning rate α balances quality/diversity).

### Weaknesses
1. The paper only mentions Kent et al. (2022)’s continuous QD Score as an “evaluation tool” but fails to deeply contrast DMS’s core innovation with Kent et al.’s static metric.\
2. DMS uses MLPs for the discount model for non-image measures) but provides no justification for why MLPs outperform other models (e.g., kernel methods, transformers for image measures). For high-dimensional image measures in LSI, the MLP’s ability to capture complex semantic relationships is unproven.
3. Uses Euclidean distance and LSI uses CLIP scores for centroid matching, but the paper does not compare these choices with alternatives (e.g., cosine similarity for images) or validate their impact on performance. 
4. The LP benchmark only tests up to 16-dimensional measure spaces. 
5. DMS adds discount model training, but the paper does not quantify its computational cost (e.g., runtime, GPU memory) relative to CMA-MAE.

### Questions
1. Could you provide a detailed comparison between DMS’s learnable continuous discount model and Kent et al. (2022)’s continuous QD Score, explaining how DMS’s model-guided exploration advances beyond the static metric in Kent et al.’s work, especially in addressing high-dimensional measure space distortion?
2. For the discount model architecture (e.g., MLPs with [k, 128, 128, 1] layers), could you provide justification for why MLPs are superior to alternative models (e.g., kernel methods for low-dimensional measures, transformers for high-dimensional image measures)?
3. In the LSI (Hiker) domain, DMS only achieves 3.77% archive coverage. Could you analyze the cause of this low coverage? Additionally, propose and test potential strategies to improve coverage.
4. The authors attribute DMS’s similar performance to CMA-MAE in TA (MNIST) to “discount model noise.” Could you quantify this noise (e.g., mean squared error between predicted and ideal discount values) and explicitly analyze how it disrupts improvement rankings and emitter updates? Also, explain why this noise has a more pronounced impact in TA (MNIST) than in other domains.
5. Could you quantify the computational overhead of DMS’s discount model training (e.g., per-iteration runtime, GPU memory usage) relative to CMA-MAE across all domains (e.g., 10D LP, LSI)? For large QDDM datasets (e.g., 10,000 LHQ images), also clarify the measures taken to ensure DMS remains computationally efficient.

### Soundness
2

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
3

### Summary
This work deals with high-dimensional quality-diversity optimisation by proposing a so-called discount model search. It aims to guide exploration
with a model that provides a smooth, continuous representation of discount scalar values.

### Strengths
The work is well-motivated - indeed, a major issue of QD is its difficulty in solving high-dimensional problems. 

The paper is well presented, and well-structured. 

The result presents the effectiveness of the proposed method.

### Weaknesses
It seems to me that a major issue is that the method needs additional information (measures) to distinguish between solutions plateaus in QD. It would be more interesting and useful if the work could directly work on algorithm design of QD itself to address the high-dimension issue, rather than resorting to new information (which apparently can tell solutions apart). I am not sure my idea is doable, but it would be very useful if it could.

### Questions
I would be grateful and may potentially increase my score if the author(s) could address my comments in the weakness field.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper addresses the challenge of "distortion" in Quality Diversity (QD) optimization, where high-dimensional measure spaces cause many solutions to map to the same regions, leading to stagnation in state-of-the-art algorithms like CMA-MAE that rely on discrete archive histograms for guiding search. The authors propose Discount Model Search (DMS), which replaces the discrete discount histogram with a continuous, learned "discount model" (a neural network) to provide smoother, more accurate improvement signals in high-dimensional spaces. Additionally, the paper introduces a new QD setting called Quality Diversity with Datasets of Measures (QDDM), allowing users to specify desired diversity via datasets (e.g., images) rather than hand-crafted functions. Experiments on standard high-dimensional benchmarks and new QDDM domains (rendering triangles to match MNIST, latent space illumination with StyleGAN3) show DMS significantly outperforms current black-box QD algorithms in terms of QD score and coverage.

### Strengths
- Originality & Significance in addressing high-dimensional measures: The core insight—replacing the discrete, low-resolution histogram of CMA-MAE with a continuous learned model—is a novel and effective solution to the well-documented problem of distortion in high-dimensional QD. This allows QD specifically to scale to measure spaces previously considered intractable for standard MAP-Elites-based approaches.

- Introduction of QDDM: The proposal of Quality Diversity with Datasets of Measures (QDDM) is a highly practical and significant contribution. By leveraging the "manifold hypothesis" and using datasets to define centroids for CVT archives, it opens QD to a vast array of new applications where hand-crafting low-dimensional measure functions is difficult, but data is abundant (e.g., using raw images as measures).

- Strong Empirical Results: DMS demonstrates superior performance across specifically designed high-distortion benchmarks (e.g., 10D Linear Projection) compared to strong baselines like CMA-MAE and DDS. The inclusion of complex, creative domains like LSI (Hiker) demonstrates practical utility beyond simple synthetic benchmarks.

- Clarity of Exposition: The paper is exceptionally clear. Figure 1 effectively illustrates the failure mode of discrete histograms in distorted spaces. The explanation of distortion and how it is amplified by dimensionality is well-reasoned.

- Thorough Ablations: The appendices contain valuable ablations, specifically demonstrating the critical necessity of "empty points" in training the discount model to prevent it from hallucinating explored regions.

### Weaknesses
- Computational Overhead: As noted by the authors, training the neural network discount model introduces computational overhead compared to the simple histogram updates of CMA-MAE. While likely acceptable given the performance gains in high dimensions, a more explicit quantification of this wall-clock time trade-off in the main text would be beneficial.

- Potential Precision Limitations: The results in the TA (MNIST) domain show DMS performing slightly (though not significantly) worse than CMA-MAE in QD Score, despite similar coverage. The authors speculate this is due to noise in the discount model interfering with fine-grained objective optimization. This suggests a potential limitation of DMS in domains where extremely precise objective optimization in already-explored regions is paramount.

- Dependence on "Empty Points": The method heavily relies on explicitly sampling "empty points" to clamp the discount model. While effective in the tested domains, one might question how this scales if the measure space is so vast that almost all points are empty. Is a fixed number $n\_{empty}$ sufficient, or does it need to scale with the measure space volume?

### Questions
1. In the QDDM setting, how does the approach scale with the size of the user-provided dataset? If a user provides a dataset of 1 million images, does constructing the CVT archive and finding nearest centroids become a bottleneck, and does DMS still effectively navigate such a large, sparsely populated space?

2. Regarding the "noise" limitation in TA (MNIST): Have you considered a hybrid approach where the discrete histogram is used for highly local, fine-grained optimization (if cell capacity allows), while the discount model guides broad exploration?

3. Could you provide more concrete details on the wall-clock time difference between DMS and CMA-MAE on the 10D benchmarks to better illustrate the computational trade-off mentioned in the conclusion?

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces Discount Model Search (DMS), a new algorithm for quality diversity (QD) optimization that addresses the challenges of exploring high-dimensional measure spaces. Traditional algorithms like CMA-MAE rely on discrete histograms to represent discount values in measure space, which causes exploration to stagnate when multiple solutions map to the same histogram cell. DMS replaces this histogram with a continuous, learned discount model—typically a neural network, that can assign smoother and more fine-grained discount values, even for similar measures.
The paper also introduces Quality Diversity with Datasets of Measures (QDDM), where the measure space is defined by a dataset (e.g., images), enabling new applications in domains like generative modeling. Extensive experiments across benchmark tasks and new QDDM domains (e.g., Triangle Arrangement and Latent Space Illumination) show that DMS outperforms state-of-the-art QD algorithms such as CMA-MAE and DDS in both QD Score and coverage.

### Strengths
- The paper tackles a well-motivated problem of QD algorithms in high-dimensional measure spaces.
- The paper is well written.
- there is extensive experimentation with diverse domains and baselines.

### Weaknesses
- If discount values of two data points are the same, then the DMS would treat them the same way. As with every quantitative measure, by Goodhart's law, pathologies may appear when we optimize for quantitative measures. Did the authors observe any of such phenomenon? How might the authors foresee mitigating such a limitation in future?

### Questions
- In Figure 2, a lot of the characteristics of the person (e.g., skin tone, facial features) remain the same even though the input image looks different. And although the image looks more similar sometimes with lakes in different images, the person looks more different. Do the authors have any insight or analysis on why this happens?
- How do the authors foresee the algorithm to be applicable to text domains? Could we reuse pretrained LLMs as the DMS?
- Is there a possibility of transferring the trained DM across tasks?
- Could the same mechanism be applied to the fitness function?

### Soundness
3

### Presentation
3

### Contribution
3
