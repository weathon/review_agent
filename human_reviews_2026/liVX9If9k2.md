# Multimodal Datasets with Controllable Mutual Information

- Avg Score: 2.50
- Decision: Reject
- Scores: 2, 2, 4, 2

## Abstract
We introduce a framework for generating highly multimodal datasets with explicitly calculable mutual information between modalities. This enables the construction of benchmark datasets that provide a novel testbed for systematic studies of mutual information estimators and multimodal self-supervised learning techniques. Our framework constructs realistic datasets with known mutual information using a flow-based generative model and a structured causal framework for generating correlated latent variables.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a framework for generating multimodal datasets where the mutual information between modalities is measurable and controllable. This is useful for lots of works which study the mutual information between modalities and labels in multimodal training dynamics.

### Strengths
- The framework for generating controllable mutual information seems correct and insightful.
- There are a lot of important use cases for this: so many multimodal works look at training based on mutual information. Having it controlled synthetically would be a powerful and useful testbed for that research, and could lead to an important breakthrough in that field.

### Weaknesses
Unfortunately, I don't think this paper did quite enough to justify that this framework could be used for the strengths I outlined above. A few key points:
- What is the practical utility of this work? You could for example show that your dataset provides training transfer to realistic environments with mutual information-dependent training methods. But without that, how do we know the value of the data you generate with your method? 
- If there isn't transfer of performance or key insights from training, what insights can you get by studying models on this dataset, and will those insights transfer to models' behavior on real world datasets? If so, this could be a useful prototyping tool that allows people to run and understand experiments theoretically before doing computationally expensive and confusing training runs on messy real world data. For example, can you show that some findings from prior work are mirrored in your setting, and can be ascertained quickly and reliably, whereas training on a full real world dataset would be costly and noisy?

### Questions
- How would you simulate handle modality imbalances? Where some data have missing modalities or you have large amounts of unimodal data.
- I didn't understand the black hole example. Could you clarify the motivation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a framework for generating synthetic data with controlled mutual information using flow-based generative models and a structured casual framework. The illustration of the data generation pipeline is followed by two brief discussions of the example usage in generating synthetic data for different underlying causal structures and scales of modalities.

### Strengths
- The paper is well-motivated, as there has been an emerging interests in multimodal learning from an information-theoretic approach, and this paper provides a well-suited, controlled testbed for such types of research;
- The proposed data generation pipeline is novel, well-documented and clearly explained;

### Weaknesses
- One major limitation of this work is the lack of empirical evaluation, neither qualitative evaluation (e.g. Figure 2, which the paper also acknowledges that "there is no clear visual connection between these pairs of images") nor quantitative evaluation. This makes it **very hard to verify the correctness** of the proposed framework. In particular, the reviewer does not agree with the claim that "our framework allows us to state unequivocally that these high-dimensional, complex datasets have a specific amount of mutual information" due to this lack of empirical evidence. The paper also does not give any empirical evaluation using the synthetic data generated from the proposed pipeline, so **the claims about the practical utility is also not testified**.

### Questions
- The reviewer strongly recommends adding more empirical evaluation of the proposed pipeline to show (1) the correctness (e.g. either qualitatively or quantitatively verify the generated data are indeed correlated by the given mutual information) and (2) the utility via a minimal set of evaluations of existing information-theoretic multimodal learning approaches on the generated data, followed by analysis on the results and potential insights that can be meaningful towards multimodal learning research from an information-theoretic perspective

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a novel framework for generating high-dimensional multimodal datasets with controllable mutual information. By using flow-based generative models, the method ensures that mutual information between latent variables is preserved, providing a theoretical foundation. The paper also designs a structured causal framework to generate correlated latent variables, derive closed-form analytical formulas for mutual information, and provide examples of synthetic multimodal datasets illustrating different causal and correlation patterns.

### Strengths
1. The paper proposes a framework for generating high-dimensional multimodal data with controllable mutual information, which is rarely achieved in existing public datasets or prior methods.

2. By leveraging flow-based generative models, the approach ensures that the generated data preserves mutual information between latent variables, providing a theoretical foundation.

### Weaknesses
1. All experiments are conducted solely on CIFAR-10 image data, without demonstrating results on real multimodal datasets (e.g., CMU-MOSI, CMU-MOSEI, or video-text-audio combinations).

2. The paper does not evaluate the generated data on downstream tasks (e.g., regression or classification), making it difficult to quantitatively assess its contribution. It also lacks direct comparison with existing mutual information estimators or multimodal SSL approaches.

3. Some concepts (e.g., the template and flow matching) are not intuitive to non-expert readers, and overall readability could be improved. Moreover, the paper is limited to 8 pages, whereas the ICLR 2026 initial submission allows up to 9 pages.

### Questions
1. How does the generated data impact performance on downstream tasks, such as regression or classification?

2. Could the authors provide a comparison of their approach with existing mutual information estimators or multimodal SSL methods to better contextualize the contributions?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes a framework for generating synthetic multimodal datasets with explicitly controllable MI between modalities. The method combines causal latent-variable construction with flow-based generative models that preserve MI under bijective mappings. The authors claim this provides a testbed for studying multimodal SSL and benchmarking mutual information estimators.

### Strengths
Although data generation is not my primary area of expertise, this work appears to address a genuinely underexplored and important problem: constructing realistic high-dimensional multimodal datasets with analytically tractable and controllable mutual information, which could enable systematic evaluation of self-supervised learning methods and mutual information estimators. The theoretical development is simple and clear. The use of flow-based generative models to maintain information structure across high-dimensional modalities is conceptually elegant and technically well-motivated.

### Weaknesses
The main limitation of this paper lies in the absence of empirical validation. While the framework is theoretically elegant, the paper does not demonstrate that the generated datasets are practically useful for their intended purposes, such as evaluating self-supervised learning methods or mutual information estimators. The examples provided are purely illustrative and rely on analytic expressions rather than experiments that confirm controllability or MI preservation in practice. Moreover, the claim of producing “realistic multimodal data” is overstated: using CIFAR-10 class-conditioned flows as a proxy for distinct modalities is a weak approximation of genuine multimodality (e.g., image–text, video–audio, etc.), and it remains unclear whether the generated samples exhibit meaningful cross-modal relationships. The reliance on linear-Gaussian causal structures, while analytically convenient, limits the generality of the approach for more complex, nonlinear dependencies in real-world multimodal settings. The paper would also benefit from quantitative experiments comparing analytical MI values with empirical estimates obtained via neural MI estimators to substantiate its proposed utility.

### Questions
1. Can you provide empirical evidence that the generated datasets preserve the specified mutual information after flow transformations?
2. Have you tested any self-supervised learning methods to demonstrate that controllable MI affects downstream performance as intended?
3. Does your linear-Gaussian setup generalize to nonlinear or non-Gaussian latent dependencies?
4. How scalable is the framework to higher-dimensional data (e.g. video or time series) or more modalities?
5. Have you evaluated how well existing mutual information estimators recover the known MI values on your generated datasets?

### Soundness
2

### Presentation
3

### Contribution
3
