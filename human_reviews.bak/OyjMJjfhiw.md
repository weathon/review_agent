# LLM Embeddings Improve Test-Time Adaptation to Tabular $Y|X$-Shifts

- Decision: Reject
- Scores: 3, 5, 3, 6, 6

## Abstract
For tabular datasets, the change in the relationship between the label and covariates ($Y|X$-shifts) is common due to missing variables. Since it is impossible to generalize to a completely new and unknown domain, we study models that are easy to adapt to the target domain even with few labeled examples. We focus on building more informative representations of tabular data that can mitigate $Y|X$-shifts, and propose to leverage the prior world knowledge in LLMs by serializing the tabular data to encode it. We find LLM embeddings alone provide inconsistent improvements in robustness, but models trained on them can be well adapted to the target domain even using 32 labeled observations. Our finding is based on a systematic study consisting of 7650 source-target pairs and benchmark against **261,000** model configurations trained by 20 algorithms. Our observation holds when ablating the size of accessible target data and different adaptation strategies.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper focuses on the test-time adaptation for tabular data with Y|X shifts. Specifically, the proposal converts the raw tabular data into a new embedding with LLM and trains a shallow neural network on the embedding. When facing the distribution shift, the authors propose to fine-tune the network for the adaptation. Experimental results on three datasets are reported.

### Strengths
1) The paper focuses on the test-time adaptation for tabular data with Y|X shift. The problem is important and has been rarely studied.
2) The proposal is clear and expected to be effective.

### Weaknesses
1) The technical contribution is limited. The paper is more like an engineering work. The adopted techniques are all widely-adopted techniques for tabular data learning and the conclusions are also not surprising. It is not clear what technical problems have been addressed or what new insights have been proposed by this paper.
2) For the experimental results, the authors only adopted three datasets and the feature dimensions are small-scale.
3) How many times about the experiments repeated and how about the performance variance?
4) The authors claim that the LLM embeddings improve performance. However, the LLM embeddings are influenced by the serialization method and the choices of different LLMs. The influence of these should be discussed.
5) It is impractical to know the distribution shift in advance. In more practical scenarios, we need to detect the distribution shift first and then adapt.

### Questions
As discussed above.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
This paper studies distribution shifts in tabular data by using embeddings from large language models (LLMs) to enhance test-time adaptation. The paper uses a LLM encoder  to featurize tabular data and fit a shallow neural network on these embeddings for tabular prediction. The paper investigates different configurations including various training algorithms, incorporating additional domain information or not, fine-tuning the shallow neural network with few target domain samples or not. The paper provides comprehensive experiments on these configurations.

### Strengths
1.  The authors provide a broad empirical evaluation across thousands of source-target pairs, diverse model configurations, and multiple algorithms, which strengthens the credibility of their findings. The paper includes a variety of performance metrics to assess the effectiveness of LLM embeddings under different conditions.  

2. By highlighting different scenarios and distributional changes, the authors clarify the conditions under which the approach excels or may face limitations. This transparency in results provides a balanced view of the method's robustness and allows researchers to see where it might or might not be effective.

3. The authors include a focused analysis on the effectiveness of few-shot adaptation, showing how minimal labeled data in the target domain can significantly improve performance.

### Weaknesses
1. The approach essentially applies existing LLMs as embeddings for tabular data without introducing substantial new methods for embedding generation or adaptation. Serializing tabular data into texts and using LLMs to extract features/make predictions have been studied in previous works. The adaptation technique described (using few-shot labeled samples for fine-tuning) is a well-known concept in transfer learning and domain adaptation. Few-shot learning techniques have been applied across various fields, and using minimal labeled data to adapt models to new domains is not unique to this work.

2. Although the paper provide comprehensive experiments on the configurations of the proposed method, it is still unclear to position how the proposed method performs in the literature. Specifically, the paper compares with baselines of Tabular Data + NN, where NN is the fully-connected neural networks. However, there are rich literatures on more advanced neural networks tailored for tabular data (e.g., FT-Transformer and many others). In addition, LLMs have been used for tabular data recently [1]. Comparisons with state-of-the-art methods on neural networks for tabular data and LLMs for tabular data are expected. Having said this, I'm not asking the authors to add more experiments in the rebuttal stage.

[1] Fang, Xi, Weijie Xu, Fiona Anting Tan, Jiani Zhang, Ziqing Hu, Yanjun Jane Qi, Scott Nickleach, Diego Socolinsky, Srinivasan Sengamedu, and Christos Faloutsos. "Large language models (LLMs) on tabular data: Prediction, generation, and understanding-a survey." (2024).

### Questions
I suggest the authors to provide a conclusion of the paper summarising the key findings and discussing the limitations.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper tackles an exciting problem by utilizing LLM embeddings for tabular classification. The authors conduct comprehensive experiments to assess the effectiveness of LLM embeddings in tabular tasks, with promising results that highlight a new and potentially impactful direction for advancing tabular data analysis.

### Strengths
1. The paper is well-structured and easy to follow, with the three technical components clearly and accessibly presented.
2. The experiments are comprehensive, which can demonstrate the claims of this paper.

### Weaknesses
1. The dataset variety is limited, as experiments are conducted on only three datasets within the same domain, which restricts the generalizability of the results and may affect the robustness of the findings.
2. The proposed method incorporates additional domain knowledge to enhance classification performance. However, it is unclear whether comparisons with tree-based methods are entirely fair, given that such methods may not effectively leverage domain knowledge. A thorough discussion on the source of performance improvements: whether from domain knowledge integration or specific technical innovations—would add valuable clarity.
3. The theoretical analysis may not be entirely appropriate for this context. Although the paper introduces a few-shot prompt tuning approach to maximize the use of limited labeled data, the theorem presented assumes a large  $m$  to achieve a small generalization bound, which contradicts the few-shot setting.
4. The term “test-time adaptation” in the title may be misleading, as it typically refers to model tuning using only unlabeled test data, which does not align with the approach described in this paper.

### Questions
Please refer to the "Weaknesses" section and answer the following questions. 
1. Why does this paper lack a conclusion section?
2. How are prompt templates tailored for different datasets? Typically, datasets with varying columns and feature types require significantly different prompts to optimize the effectiveness of the LLM.

### Soundness
2

### Presentation
3

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
This paper proposes a method to solve the Y|X shift problem in the tabular data domain using LLM embeddings. Under the LLM+NN model architecture, it uses concatenation to combine the domain information of LLM with the original tabular data information for improving prediction robustness and performance. The paper study models that are easy to adapt to the target domain even with few labeled examples so that fine-tuning the classification model with a small number of samples can further improve performance. Experiments can prove the effectiveness of the proposed method.

### Strengths
1. The motivation is clear and the algorithm is sensible.
2. The proposed method is tested on several benchmarks.
3. The proposed method is easy to understand and simple. It does not require a large number of LLM calls and the computational complexity is not high.

### Weaknesses
1. On adaptation stage: The method requires the target's true label to guide the model parameter update. This setting is different from the basic test-time adaptation (TTA) setting, which is generally an unsupervised objective function [1]. The setting in this article is more like a fine-tuning setting.
2. The experiments in this article are mainly based on the ACS dataset. In recent studies, there are some other benchmarks like TableShift [2] contains Y|X shifts datasets. In the report they provided, ACS Income and ACS Pub.Cov dataset seem not have severe Y|X shifts. In order to prove the robustness to severe shifts, I think experiments on those datasets like ACS Unemployment and ICU Length of Stay are necessary.

[1] Wang, Dequan, et al. "Tent: Fully test-time adaptation by entropy minimization." arXiv preprint arXiv:2006.10726 (2020).

[2] Gardner, Josh, Zoran Popovic, and Ludwig Schmidt. "Benchmarking distribution shift in tabular data with tableshift." Advances in Neural Information Processing Systems 36 (2024).

### Questions
1. Figure 3: Recent study shows that label shift in tabular data causes performance degrade [1], why not take label shift into account? 
2. Are Y|X shifts and concept shift the same in meaning?
3. How does proposed compare to state-of-the-art methods designed for tabular TTA [2]?
4. How to evaluate Y|X shifts degree?

[1] Gardner, Josh, Zoran Popovic, and Ludwig Schmidt. "Benchmarking distribution shift in tabular data with tableshift." Advances in Neural Information Processing Systems 36 (2024).

[2] Ren, Weijieying, et al. "TabLog: Test-Time Adaptation for Tabular Data Using Logic Rules." Forty-first International Conference on Machine Learning.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the challenges posed by shifts in tabular data, specifically shifts in the relationship between labels and covariates (Y|X-shifts). These shifts often occur due to missing variables or confounders, making it difficult for models trained on a source domain to generalize to a target domain. 

The authors propose a solution using embeddings generated from large language models (LLMs) to encode tabular data. They demonstrate that LLM embeddings can improve adaptation to target domains even with limited labeled target samples, as the embeddings incorporate a broader contextual knowledge that helps mitigate the effects of these Y|X-shifts. Their study includes a comprehensive evaluation across 7,650 source-target pairs and comparisons with 20 algorithms, revealing that shallow neural networks (NNs) trained on LLM embeddings outperform other models in many cases.

The study's results highlight that while LLM embeddings alone do not consistently surpass state-of-the-art tree-based models, their performance significantly improves when adapted with a small number of target samples. Specifically, using 32 labeled target samples for fine-tuning yields substantial gains across multiple datasets, especially under scenarios of strong Y|X-shifts. The authors also explore different methods of adaptation, including in-context learning, low-rank adaptation, and prefix tuning, finding that fine-tuning with LLM embeddings provides a practical approach for generalizing across tabular Y|X-shifts.

### Strengths
This paper excels in addressing a pressing issue in machine learning: adapting models to distributional shifts in tabular data, specifically Y|X-shifts. The authors take an innovative approach by leveraging LLM embeddings to encode tabular data, which allows the model to incorporate broader contextual knowledge. This strength is significant because it enables the model to adapt to changes in label-covariate relationships, even with minimal (32 examples) labeled data from the target domain. The comprehensive evaluation of 7,650 source-target pairs across three datasets (ACS Income, Mobility, and Public Coverage) highlights the robustness of their approach. By testing against 20 algorithms (traditional GBDT and NN based models) and exploring a vast configuration space, the authors provide a solid foundation for the efficacy of LLM embeddings in generalizing across diverse tabular data environments. This thorough evaluation not only strengthens the empirical findings but also sets a new benchmark in evaluating distributional shifts in tabular datasets.

### Weaknesses
This method is only applicable when there is a description available for all features of the tabular data. It would be good to expand this method to datasets which include features without any description. A possible solution can be by learning embeddings of such features from scratch and utilizing them as is done in TabTransformer. This will provide a comprehensive solution for different types of tabular data.

### Questions
The proposed method appears limited to datasets where all features have available descriptions. Expanding this approach to include datasets with features lacking descriptions would increase its versatility. One possible solution could involve learning embeddings for such features from scratch, similar to methods like TabTransformer, which can effectively handle mixed data types by embedding categorical features without requiring explicit descriptions.

It would be beneficial for the authors to discuss this limitation in the manuscript and potentially compare their method to approaches like TabTransformer that support feature embedding without descriptions. This addition would provide a clearer understanding of the method’s scope and suggest pathways for future improvements in handling various types of tabular data.

### Soundness
3

### Presentation
3

### Contribution
3
