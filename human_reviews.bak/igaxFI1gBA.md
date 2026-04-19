# Temporal Causal Discovery and Generative Prediction of Vehicular CO$_2$ emission

- Decision: Reject
- Scores: 3, 6, 5

## Abstract
Global warming from greenhouse gas emissions is humanity's largest environmental hazard. Greenhouse gases, like CO$_2$ emissions from transportation, notably cars, contribute to the greenhouse effect. Effective CO$_2$ emission monitoring is needed to regulate vehicle emissions. Few studies have predicted automobile CO$_2$ emissions using OBD port data. For precise and effective prediction, the system must capture the underlying cause-effect structure between vehicular parameters that may contribute to the emission of CO$_2$ in the transportation sector. Thus, we present a causal RNN-based generative deep learning architecture that predicts vehicle CO$_2$ emissions using OBD-II data while keeping the underlying causal structure. Most widely used real-life datasets lack causal relationships between features or components, so we use our proposed architecture to discover and learn the underlying causal structure as an adjacency matrix during training and employ that during forecasting. Our framework learns a sparse adjacency matrix by imposing a sparsity-encouraging penalty on model weights and allowing some weights to be zero. This matrix is capable of capturing the causal relationships between all variable pairs. In this work, we first train the model with widely used synthetic datasets with known causal structure among variables, then we apply it to the state-of-the-art OBD-II dataset to find the internal causal structure among the vehicular parameters and perform causal inference to predict CO$_2$ emission. Experimental results reveal that our causal discovery and forecasting method surpasses state-of-the-art methods for the tasks of causal discovery in terms of AUROC, forecasting on multivariate causal time series data, and OBD-II dataset in terms of MMD, RMSE, and MAE. After successful completion, we will release the code (Code for review - \href{https://anonymous.4open.science/r/causal-obd-co2-0A0C}{https://anonymous.4open.science/r/causal-obd-co2-0A0C}).

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper presents a new framework for predicting and discovering the casual structure of vehicular emission by learning a sparse adjacency matrix.

### Strengths
1.	The motivation of the paper is clearly presented, and the writing is easy to follow. 
2.	The proposed method is tested on both real-world and synthetic datasets.

### Weaknesses
1.	Although the main focus of this paper is casual discovery, the casual discovery performance of the proposed method has not been tested on any real-world datasets.
2.	Some important implementation details are missing. For example, what are the exact form for the regularization term R in Eq. 7 and the forecasting lengths, receptively?   
3.	The experimental results are weak. The proposed method should be compared to more state-of-the-art causal discovery methods in the field. Below are some suggestions:
a) Gao, Shanyun, Raghavendra Addanki, Tong Yu, Ryan Rossi, and Murat Kocaoglu. "Causal discovery in semi-stationary time series." Advances in Neural Information Processing Systems 36 (2023).
b) Löwe, Sindy, David Madras, Richard Zemel, and Max Welling. "Amortized causal discovery: Learning to infer causal graphs from time-series data." In Conference on Causal Learning and Reasoning, pp. 509-525. PMLR, 2022.
c) Gerhardus, Andreas, and Jakob Runge. "High-recall causal discovery for autocorrelated time series with latent confounders." Advances in Neural Information Processing Systems 33 (2020): 12615-12625.
4.	As a generative time series model, the proposed method also should be compared with state-of-the-art diffusion-based models, such as
a)	Tashiro, Yusuke, Jiaming Song, Yang Song, and Stefano Ermon. "Csdi: Conditional score-based diffusion models for probabilistic time series imputation." Advances in Neural Information Processing Systems 34 (2021): 24804-24816.

### Questions
1.	Why the results presented in Fig. 4 and 5 are used to indicate the effectiveness of the proposed method? The time steps of the data sample are different.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The author proposes a recursive variational autoencoder (VAE) to approximate the joint distribution of time series data and improve the interpretability of time series forecasting by learning Granger causal structures. This framework also handles causal relationships within multivariate time series data and benefits future time step predictions through a generative model.

### Strengths
This paper offers an insightful approach by incorporating causal inference to enhance multivariate time series generation. For instance, this work could inspire future research in few-shot scenarios or cases where certain variables are sparse in multivariate time series generation. Additionally, it could delve deeper into identifying which variables contribute to long-term sequence modeling and which are more effective in capturing short-term temporal changes. I believe this paper has the potential to inspire a range of related future work.

### Weaknesses
1.Formatting Issue:
 Line 121 is missing a space.

2.Writing Issue:
First, the motivation for the work—improving multivariate time series forecasting by better learning causal relationships—has not been clearly communicated to the reader. It is important to show the potentially causal variables in the context of the CO2 forecasting problem. If text space is a concern, these details should be included in the appendix. 
Additionally, if you can provide a brief explanation or demonstration of why understanding the intrinsic causal relationships in multivariate data leads to better generative time series forecasting, it would be a significant enhancement to the paper. Your current equations are close to achieving this, but the argument needs a bit more development.

3.Experimental Issue:
For the generative forecasting experiments, the authors compare TGAN (2019), VRNN (2015), and VRAE (2013). The choice of these baselines demonstrates the authors’ thoughtful consideration by directly comparing their work to the most relevant generative forecasting methods. However, in recent years, several emerging approaches for multivariate time series generative forecasting have been proposed, such as ScoreGrad (2021) and D3VAE (NeurIPS, 2022).
For simple RNN-based forecasting, I find that including only one RNN model as a baseline for non-generative multivariate time series modeling is insufficient. More recent and effective models should be considered in this context. Given the standards of ICLR, the experimental setup presents notable shortcomings.

### Questions
1. A statement is not strict enough：
The claim that the work you're referring to is the first to use a recurrent variational autoencoder (VAE) for time series forecasting is not entirely accurate. Previous research has already explored using Variational Recurrent Autoencoders (VRAE) for time series data. One notable example is the VRAE model (2014), VRNN (2015). The novelty of your mentioned work could lie in the specific architecture or methodology used, such as the incorporation of causal discovery through Granger causality and the multi-head decoder design, which may not have been part of earlier models. Could you further improve your claim about this?
2. Could you attempt to briefly demonstrate or explain why a better understanding of the intrinsic causal relationships within multivariate data can lead to improved multivariate time series generation?
Since the experimental results are not particularly strong, clarifying the rationale and theory behind your approach would help improve the overall quality and coherence of the paper.
3. As for supplementing the experiments, I recommend following the suggestions and opinions of other reviewers, especially about the multivariate time series model forecasting besides RNN.  I may not increase or decrease the score based on whether additional experiments are included in rebuttal phase.  The second item is more important for me.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a causal RNN-based generative deep learning architecture that predicts vehicle CO2 emissions using OBD-II data while maintaining the underlying causal structure. Unlike traditional models that miss hidden cause-effect relationships, this framework discovers and learns these relationships as an adjacency matrix during training by applying a sparsity-encouraging penalty. It captures causal links between variables and employs this knowledge for forecasting.

### Strengths
1. The background of this work is clearly presented, emphasizing that controlling CO2 emissions is a critical issue that needs to be addressed.
2. This paper designs and conducts two distinct experimental setups for causal discovery and emission forecasting, along with comprehensive result analysis.
3. The paper provides open-source code to enable the replication of its results.

### Weaknesses
1. The paper is not well written. The introduction does not summarize the contributions of this work. The importance of causal discovery for emission forecasting needs further clarification, including why causality between time series is significant for emission forecasting. For example, the causal analysis for traffic forecasting can help traffic managers know the reason for the jam happening.

2. The proposed method lacks novelty. The approach does not include designs specific to CO2 characteristics and relies solely on existing RNN-based architectures. Why do current approaches for time series forecasting struggle with accurately predicting CO2 emissions?

3. The selection of baselines for the forecasting task is insufficient, as it lacks recent state-of-the-art methods from the past three years. 

4. While causal discovery appears effective, it lacks validation with real-world data. Case studies are needed to demonstrate its practical application. You can do case studies on the real-world dataset you provide, and show the discovery results.

### Questions
Same as the weakness.

### Soundness
3

### Presentation
3

### Contribution
3
