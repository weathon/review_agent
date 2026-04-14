

{0}------------------------------------------------

# EXPAND AND COMPRESS: EXPLORING TUNING PRINCIPLES FOR CONTINUAL SPATIO-TEMPORAL GRAPH FORECASTING

Wei Chen, Yuxuan Liang<sup>✉</sup>

The Hong Kong University of Science and Technology (Guangzhou)

onedeanxxx@gmail.com, yuxliang@outlook.com

## ABSTRACT

The widespread deployment of sensing devices leads to a surge in data for spatio-temporal forecasting applications such as traffic flow, air quality, and wind energy. Although spatio-temporal graph neural networks (STGNNs) have achieved success in modeling various static spatio-temporal forecasting scenarios, real-world spatio-temporal data are typically received in a streaming manner, and the network continuously expands with the installation of new sensors. Thus, spatio-temporal forecasting in streaming scenarios faces dual challenges: the inefficiency of retraining models over newly-arrived data and the detrimental effects of catastrophic forgetting over long-term history. To address these challenges, we propose a novel prompt tuning-based continuous forecasting method, EAC, following two fundamental tuning principles guided by empirical and theoretical analysis: *expand and compress*, which effectively resolve the aforementioned problems with lightweight tuning parameters. Specifically, we integrate the base STGNN with a continuous prompt pool, utilizing stored prompts (*i.e.*, few learnable parameters) in memory, and jointly optimize them with the base STGNN. This method ensures that the model sequentially learns from the spatio-temporal data stream to accomplish tasks for corresponding periods. Extensive experimental results on multiple real-world datasets demonstrate the multi-faceted superiority of EAC over the state-of-the-art baselines, including effectiveness, efficiency, universality, etc. Our code repository is available at <https://github.com/Onedean/EAC>.

## 1 INTRODUCTION

Spatio-temporal data is ubiquitous in various applications, such as traffic management (Avila & Mezić, 2020), air quality monitoring (Liang et al., 2023), and wind energy deployment (Yang et al., 2024). Spatio-temporal graph neural networks (STGNNs) (Jin et al., 2023; 2024) have become a predominant paradigm for modeling such data, primarily due to their powerful spatio-temporal representation learning capabilities, which consider the both spatial and temporal dimensions of data by learning temporal representations of graph structures. However, most existing works (Li et al., 2017; Wu et al., 2019; Bai et al., 2020; Cini et al., 2023; Han et al., 2024) assume a static setup, where STGNN models are trained on the entire dataset over a limited time period and maintain fixed parameters after training is completed. In contrast, real-world spatio-temporal data (Liu et al., 2024; Yin et al., 2024) typically exists in a streaming format, with the underlying network structure expanding through the installation of new sensors in surrounding areas, resulting in a constantly evolving spatio-temporal network. *Due to computational and storage costs, it is often impractical to store all data and retrain the entire STGNN model from scratch for each time period.*

To address this problem, several straightforward solutions are available, as illustrated in Figure 1. The simplest approach involves *pre-training* an STGNN (using node-count-free graph convolution operators) for testing across subsequent periods. However, due to distribution shifts (Wang et al., 2024a), this method often fails to adapt to new period data. Another approach involves model *retraining* and prediction on different data windows due to graph expansion. Unfortunately, this neglects the informational gains from historical data, leading to limited performance improvements. To simultaneously resolve the challenges posed by these two issues, a more effective solution is to adopt a continual learning paradigm (Wang et al., 2024d), which is a research area focused on how

{1}------------------------------------------------

![Figure 1: Comparison of classic schemes and EAC for continual spatio-temporal forecasting. The diagram illustrates four training paradigms: Pre-training, Re-training, Online-training, and Ours. Above them, a timeline shows 'Period #1 Data', 'Period #2 Data', ..., 'Period #N Data' being processed sequentially. Pre-training shows models #1, #2, ..., #N being trained on their respective periods. Re-training shows models #1, #2, ..., #N being re-trained on all previous periods. Online-training shows models #1, #2, ..., #N being trained on new periods while freezing previous ones. Ours shows a continuous prompt pool (orange) being updated and used across all periods, with models #1, #2, ..., #N being frozen and re-loaded as needed.](9ba3dc91984c80b96f217fb1bddd5c06_img.jpg)

Figure 1: Comparison of classic schemes and EAC for continual spatio-temporal forecasting. The diagram illustrates four training paradigms: Pre-training, Re-training, Online-training, and Ours. Above them, a timeline shows 'Period #1 Data', 'Period #2 Data', ..., 'Period #N Data' being processed sequentially. Pre-training shows models #1, #2, ..., #N being trained on their respective periods. Re-training shows models #1, #2, ..., #N being re-trained on all previous periods. Online-training shows models #1, #2, ..., #N being trained on new periods while freezing previous ones. Ours shows a continuous prompt pool (orange) being updated and used across all periods, with models #1, #2, ..., #N being frozen and re-loaded as needed.

Figure 1: Comparison of classic schemes and EAC for continual spatio-temporal forecasting.

systems learn sequentially from continuous related data streams. Specifically, the core idea is to load the previously trained model for new period data and conduct *online training* on the current period. *Nevertheless, the notorious problem of catastrophic forgetting (van de Ven et al., 2024) in neural networks often hinders the improvement of online learning performance..*

Current methods for continual spatio-temporal forecasting typically follow various types of continual learning approaches for improvement. For example, TrafficStream (Chen et al., 2021b) comprehensively integrates regularization and replay-based methods to learn and adapt to ongoing data streams while retaining past knowledge to enhance performance. PECPM (Wang et al., 2023b) and STKEC (Wang et al., 2023a) further refine replay strategies to detect stable and changing node data for better adaptation. TFMoE (Lee & Park, 2024) method considers training a sets of mixture of experts models for adapting to new nodes, thereby improving efficiency. *Though promising, the aforementioned methods still involve optimizing the entire STGNN model, resulting in complex tuning costs and failing to mitigate the problem of catastrophic forgetting in a principle way.*

To this end, we propose EAC, a novel continual spatio-temporal graph forecasting framework based on *a continuous prompt parameter pool* for modeling streaming spatio-temporal data. Specifically, we freeze the base STGNN model during the continual learning process to prevent knowledge forgetting, adapting solely through a dynamically adjustable prompt parameter pool to accommodate the continuously emerging expanded node data while further storing the knowledge acquired from streaming spatio-temporal data. Notably, we explore *two fundamental tuning principles*, expand and compress, through empirical and theoretical analyses to balance model effectiveness and efficiency. EAC has five distinctive characteristics: (i) *Simplicity*: it accomplishes complex continual learning tasks solely by tuning the prompt parameter pool; (ii) *Effectiveness*: it demonstrates consistent performance across multiple real-world datasets; (iii) *Universality*: it demonstrates consistent performance across different STGNN architectures; (iv) *Efficiency*: it accelerates model training speed by effectively freezing the backbone model; and (v) *Lightweight*: it requires adjustment of only a limited number of parameters in the prompt pool. In summary, our main contributions are:

- We propose a prompt-based continual spatio-temporal forecasting framework EAC that is simple, effective, and efficient with lightweight tunable parameters.
- Through empirical observations and theoretical analysis, we explore two tuning principles for continual spatio-temporal forecasting: the heterogeneity property for expansion and low-rank property for compression in our continuous prompt parameter pool.
- Based on the two proposed tuning principles, we introduce two implementation schemes: continuous prompt pool growth and continuous prompt pool reduction.
- Experimental results on different scenarios of real-world datasets from different domains demonstrate the effectiveness and universal superiority of EAC.

## 2 RELATED WORK

**Spatio-temporal Forecasting.** Spatio-temporal forecasting originates from time series analysis and can be viewed as a temporal data modeling problem within an underlying network. Traditional statistical models, such as ARIMA (Box & Pierce, 1970) and VAR (Biller & Nelson, 2003), as well

{2}------------------------------------------------

as advanced time series deep learning models (Nie et al., 2023; Wu et al., 2022), can simplify this into a single time series forecasting task. Even though, these methods fail to capture spatio-temporal correlations between different locations, leading to suboptimal performance. STGNNs, due to their inherent ability to aggregate local spatio-temporal information, are considered powerful tools for modeling this data. STGNNs consist of two key components: a graph operator module for spatial relationship modeling, typically categorized into spectral GNNs (Yu et al., 2017), spatial GNNs (Li et al., 2017), or hybrids, and a sequence operator module for temporal relationship modeling, which can be recurrent-based (Pan et al., 2019), convolution-based (Wu et al., 2019), attention-based (Guo et al., 2019), or a combination of these networks. *However, most spatio-temporal graph forecasting models focus on static settings with limited-period forecasting scenarios.*

**Continual Learning.** Continual learning is a technique for sequentially training models as data from related tasks arrives in a streaming manner. Common approaches include regularization-based (Kirkpatrick et al., 2017), replay-based (Rolnick et al., 2019), and prototype-based (De Lange & Tuytelaars, 2021) methods, all aimed at learning knowledge from new tasks while retaining knowledge from previously tasks. However, these methods primarily focus on vision and text domains (Wang et al., 2024d), assuming that samples are *i.i.d.* TrafficStream (Chen et al., 2021b) first integrates the ideas of regularization and replay into continual spatio-temporal forecasting scenarios. PECPM (Wang et al., 2023b) and STKEC (Wang et al., 2023a) further incorporate prototype-based ideas for enhancement. TFMoE (Lee & Park, 2024) advances replay data into a generative reconstruction approach, equipped with a mixture of experts model. Additionally, some methods consider diverse perspectives such as few-shot scenarios (Wang et al., 2024b), large-scale contexts (Wang et al., 2024c), and combinations with reinforcement learning (Xiao et al., 2022) and data augmentation (Miao et al., 2024). *Nonetheless, most methods in dynamic scenarios still require tuning all STGNN parameters, resulting in the dual challenges of catastrophic forgetting and inefficiency.*

**Prompt Learning.** Prompt learning suggest simply tuning frozen language or vision models to perform downstream tasks by learning prompt parameters attached to the input to guide model predictions. Some studies (Yuan et al., 2024; Li et al., 2024) attempt to integrate it into spatio-temporal forecasting, but they still focus on static scenarios. Other methods have applied it to continual learning contexts; however, they only concentrate on vision (Wang et al., 2022) and text (Razdaibiedina et al., 2023) domains. In contrast to the various prompt-based approaches in the existing literature, a naive application of prompt learning in our context is to append learnable parameters  $P$  (referred to as prompts) to the original spatio-temporal data  $X$ , resulting in a fused embedding  $X' = [X \parallel P]$ , which is then fed into the base STGNN model  $f_\theta(X')$  for spatio-temporal forecasting. *Notably, we design a novel prompt pool learning mechanism, guided by two tuning principles derived from empirical and theoretical analysis, to model continual spatio-temporal forecasting.*

## 3 PRELIMINARIES

**Definition (Dynamic Streaming Spatio-temporal Graph).** We consider a dynamic streaming spatio-temporal graph  $\mathbb{G} = (\mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_T)$ , for every time interval  $\tau$ , the network dynamically grows, *i.e.*,  $\mathcal{G}_\tau = \mathcal{G}_{\tau-1} + \Delta\mathcal{G}_\tau$ . Specifically, the network in the  $\tau$ -th time interval is modeled by the graph  $\mathcal{G}_\tau = (\mathcal{V}_\tau, \mathcal{E}_\tau, A_\tau)$ , where  $\mathcal{V}_\tau$  is the set of nodes corresponding to the  $|\mathcal{V}_\tau| = n$  sensors in the network, and  $\mathcal{E}_\tau$  signifies the edges connecting the node set, which can be further represented by the adjacency matrix  $A_\tau \in \mathbb{R}^{n \times n}$ . The node features are represented by a three-dimensional tensor  $X_\tau \in \mathbb{R}^{n \times t \times c}$ , denoting the  $c$  features of the records of all  $n$  nodes observed on the graph  $\mathcal{G}_\tau$  in the past  $t$  time steps. Following (Chen et al., 2021b),  $c$  here is usually only a numerical value.

**Problem (Continual Spatio-temporal Graph Forecasting).** The continual spatio-temporal graph forecasting can be viewed as learning the optimal prediction model for the current stage from dynamic streaming spatio-temporal graph data. Specifically, given the training data  $\mathcal{D} = \{D_\tau | (\mathcal{G}_\tau, X_\tau, Y_\tau)\}_{\tau=1}^T \sim \mathcal{P}$  from a sequence of streaming data, our goal is to incrementally learn the optimal model parameters  $f_{\theta^*}$  from the sequential training set. For the current  $\tau$ -th time interval, the model is optimized to minimize:

$$f_{\theta^{(\tau)*}} = \operatorname{argmin}_{\theta^{(\tau)}} \mathbb{E}_{D_\tau \sim \mathcal{P}^{(\tau)}} [\mathcal{L}(f_{\theta^{(\tau)}}(\mathcal{G}_\tau, X_\tau), Y_\tau)], \quad (1)$$

where  $f_{\theta^{(\tau)*}}$  represents the optimal model that achieves the minimum loss when trained on the data from the current period  $\tau$ . The loss function  $\mathcal{L}(\cdot)$  measures the discrepancy between the predicted signals  $\hat{Y}_\tau = f_{\theta^{(\tau)*}}(\mathcal{G}_\tau, X_\tau) \in \mathbb{R}^{n \times t \times c}$  for next  $t$  time steps and the ground-truth  $Y_\tau$ .

{3}------------------------------------------------

## 4 METHODOLOGY

In this section, we propose two tuning principles, *expand* and *compress*, through detailed empirical observations and theoretical analysis. Based on these principles, we apply them to a prompt parameter pool to develop the continual spatio-temporal graph forecasting framework EAC (as shown in Figure 2). Specifically, we design a node-level prompt parameter pool corresponding to the input spatio-temporal data of different nodes, jointly optimized within the STGNN backbone. ❶ For the expand process, empirical studies reveal that the prompt parameter pool adapts to dynamic heterogeneity, which we further analyze theoretically. Building on this, we show that expanding prompt parameters for newly introduced nodes effectively accommodates heterogeneity in continuous spatio-temporal scenarios. ❷ For the compress process, empirical results indicate that the prompt parameter pool exhibits a low-rank property, which we formalize through detailed analysis. Based on this, we show that high-dimensional prompt parameters can be compressed into two low-dimensional components, mitigating parameter inflation caused by expansion in continuous spatio-temporal scenarios. We also summarize the workflow of EAC in Algorithm 1, and provide a detailed explanation of the continual learning process in Appendix B.

![Figure 2: The overall architecture of our proposed EAC. The diagram illustrates the continual learning process over multiple periods (Period #1, Period #2, ..., Period #N) over time. In each period, a graph is expanded with new nodes. The training process involves a 'Continual Training' phase where the model is updated with new data (X^t, G^t). Below this, the 'Prompt Parameter Pool' is shown, which is updated via 'Reload Weights'. The 'STGNN Backbone' is also shown, with 'Tuning Principle' steps involving 'Prompt Expand' and 'Prompt Compress' operations. A legend on the right identifies symbols: Stable Node (green circle), New Node (orange circle), Tuning (flame), Frozen (snowflake), and Prompt (orange rectangle).](d3ca266c298aeb34b019960c6c36f187_img.jpg)

Figure 2: The overall architecture of our proposed EAC. The diagram illustrates the continual learning process over multiple periods (Period #1, Period #2, ..., Period #N) over time. In each period, a graph is expanded with new nodes. The training process involves a 'Continual Training' phase where the model is updated with new data (X^t, G^t). Below this, the 'Prompt Parameter Pool' is shown, which is updated via 'Reload Weights'. The 'STGNN Backbone' is also shown, with 'Tuning Principle' steps involving 'Prompt Expand' and 'Prompt Compress' operations. A legend on the right identifies symbols: Stable Node (green circle), New Node (orange circle), Tuning (flame), Frozen (snowflake), and Prompt (orange rectangle).

Figure 2: The overall architecture of our proposed EAC .

### 4.1 EXPAND: HETEROGENEITY-GUIDED CONTINUOUS PROMPT POOL GROWTH

**Insight.** As mentioned above, fine-tuning an existing STGNN model with new data streams often leads to catastrophic forgetting (van de Ven et al., 2024). While previous methods have proposed some mitigative strategies (Chen et al., 2021b; Wang et al., 2023a;b), these solutions are not entirely avoidable. A straightforward solution is to isolate parameters, freeze the old model, and dynamically adjust the network structure to incorporate adaptable learning parameters. Recently, there has been an increasingly common consensus in spatio-temporal forecasting to introduce node-specific trainable parameters as spatial identifiers to achieve higher performance (Shao et al., 2022; Liu et al., 2023; Dong et al., 2024; Yeh et al., 2024). Although some empirical evidence (Shao et al., 2023; Cini et al., 2024) supports their predictive performance in static scenarios, there has been no root analysis to explain why they are useful, when they are applicable, and in what contexts they are most suitable. However, we find this closely aligns with our motivation and extend it to the continual spatio-temporal forecasting setting by providing a reasonable explanation from the perspective of heterogeneity to address these questions. Specifically, spatio-temporal data generally exhibit two characteristics: *correlation* and *heterogeneity* (Geetha et al., 2008; Wang et al., 2020). The former is naturally captured by various STGNNs, as they automatically aggregate local spatial and temporal information. However, given the message-passing mechanism of STGNNs, the latter is clearly not captured. Therefore, we argue that the introduction of node prompt parameter pool likely enhances the model’s ability to capture heterogeneity by expanding the expressiveness of the feature space.

**Empirical Observation.** To quantitatively analyze heterogeneity, we consider the dispersion of node feature vectors in the feature space (Fan et al., 2024). We first define the *Average Node Deviation* ( $D(\cdot)$ ) metric as:  $D(X) = \frac{1}{n \times n} \sum_{i=1}^n \sum_{j=1}^n \sum_{k=1}^d (X_{ik} - X_{jk})^2$ , where  $X \in \mathbb{R}^{n \times d}$  represent the feature matrix composed of  $n$  node vectors, each with  $d$  dimensions. This metric quantifies the degree of dispersion between pairs of node vectors within the feature matrix, reflecting the ability to express heterogeneity. We use this indicator to plot the dispersion degree of the feature matrix

{4}------------------------------------------------

for the pems-stream dataset across different periods as node prompt parameters are injected during the learning process. As shown in Figure 3, two phenomena are clearly observed: ❶ Within the same period, the dispersion of the node feature space continuously expands throughout the learning process, reflecting the increasing ability of prompt parameters to represent heterogeneity. ❷ Across different periods, the dispersion of the feature space in the current period expands further compared to the previous period, showing the continuous capture ability of the prompt parameter for heterogeneity.

**Theoretical Analysis.** Below, we provide a theoretical analysis of the above empirical results.

**Proposition 1.** For an original node input feature matrix  $X = [x_1, \dots, x_n] \in \mathbb{R}^{n \times d}$ , we introduce a node prompt parameter matrix  $P = [p_1, \dots, p_n] \in \mathbb{R}^{n \times d}$ . Through a spatio-temporal learning function  $f_\theta$  with invariance, a new feature matrix  $X^\theta = f(\theta; X, P)$  is obtained, satisfying:

$$D(X^\theta) - D(X) = 2\left(\frac{1}{n} \sum_{i=1}^n \|p_i^\theta\|^2 - \|\mu_p^\theta\|^2\right) \geq 0, \quad (2)$$

where  $P^\theta = [p_1^\theta, \dots, p_n^\theta] \in \mathbb{R}^{n \times d}$  represents the optimized prompt parameter matrix, and  $\mu_p^\theta = \frac{1}{n} \sum_{i=1}^n p_i^\theta$  is the mean vector of the parameter matrix.

*Proof.* For more details, refer to the supplementary materials in appendix A.1.  $\square$

#### **Tuning Principle I: Prompt Parameter Pool Can Continuously Adapt to Heterogeneity Property.**

**Implementation Details.** Based on the above analysis, we present the implementation details for expand process in continuous spatio-temporal forecasting scenarios. Specifically, we continuously maintain a prompt parameter pool  $\mathcal{P}$ . For the initial static stage, we provide each node with a learnable parameter vector, and the matrix  $P^{(1)}$  of all such vectors is added to the parameter pool  $\mathcal{P} = [P^{(1)}]$ . Follow the Occam’s razor, we adopt a simple yet effective fusion method, where the prompt pool and the corresponding input node features are added element-wise. The prompt parameter pool  $\mathcal{P}$  is then trained together with the base STGNN model. For subsequent period  $\tau$ , we only provide prompt parameter vectors for newly added nodes, and the resulting matrix  $P^{(\tau)}$  is added to the prompt pool  $\mathcal{P} = [P^{(1)}, P^{(2)}, \dots, P^{(\tau-1)}]$ . As we analyzed, we freeze the STGNN backbone and only tuning the prompt pool  $\mathcal{P}$ , effectively reducing computational costs and accelerate training.

### 4.2 COMPRESS: LOW-RANK-GUIDED CONTINUOUS PROMPT POOL REDUCTION

**Insight.** While node-customized prompt parameter pools are highly effective, an unavoidable challenge arises in our scenario of continuous spatio-temporal forecasting: the number of prompt parameters continuously increases with the addition of new nodes across consecutive periods, leading to parameter inflation. Despite the existence of numerous well-established studies that enhance the efficiency of spatio-temporal prediction (Bahadori et al., 2014; Yu et al., 2015; Chen et al., 2023; Ruan et al., 2024) and imputation (Chen et al., 2020; 2021a; Nie et al., 2024) tasks using techniques such as compressed sensing and matrix / tensor decomposition, these study typically focus solely on the original spatio-temporal data. An intuitive solution is to similarly apply low-rank matrix approximations to the prompt learning parameter pool, thereby reducing the number of learnable parameters while maintaining performance. However, for the prompt learning parameter pool, it remains to be validated whether it exhibits redundancy characteristics akin to spatio-temporal data and whether these properties hold in the continuous spatio-temporal forecasting setting.

**Empirical Observation.** To explore redundancy, we conduct a spectral analysis of the prompt parameter pool  $\mathcal{P}$ . Specifically, for the models optimized annually on the PEMS-Stream dataset, we first apply singular value decomposition to the extended prompt parameter pool introduced in the

![Figure 3: Heterogeneity measurement. A line graph titled 'PEMS-Stream' showing 'Average Node Deviation' on the y-axis (0 to 40) versus 'Steps of saving models' on the x-axis (0 to 16). Data series are shown for years 2011 (blue circles), 2012 (orange squares), 2013 (green triangles), 2014 (red diamonds), 2015 (purple inverted triangles), 2016 (grey diamonds), and 2017 (pink stars). All series show an upward trend in average node deviation as the number of steps increases, with the 2017 series reaching the highest deviation of approximately 40 at step 16.](867fce43c58fda6178b06e454b4ed73a_img.jpg)

Figure 3: Heterogeneity measurement. A line graph titled 'PEMS-Stream' showing 'Average Node Deviation' on the y-axis (0 to 40) versus 'Steps of saving models' on the x-axis (0 to 16). Data series are shown for years 2011 (blue circles), 2012 (orange squares), 2013 (green triangles), 2014 (red diamonds), 2015 (purple inverted triangles), 2016 (grey diamonds), and 2017 (pink stars). All series show an upward trend in average node deviation as the number of steps increases, with the 2017 series reaching the highest deviation of approximately 40 at step 16.

Figure 3: Heterogeneity measurement.

{5}------------------------------------------------

previous section and plot the normalized cumulative singular values for different years, as shown in Figure 4 left. It can be observed that ❶ all years exhibit a clear long-tail spectral distribution, indicating that most information from the parameter matrix  $\mathcal{P}$  can be recovered from the first few largest singular values. In Figure 4 right, we also present a heatmap of the normalized cumulative singular values at the sixth largest singular value for different years at different steps, revealing that, ❷ despite some variations across years, the overall processes for all years maintain a high concentration of information ( $> 0.75$ ), suggesting a low-rank property for  $\mathcal{P}$ .

![Figure 4: Low-rank measurement. The figure consists of two parts. The left part is a line plot titled 'PEMS-Stream' showing 'Normalized Cumulative Singular Value' on the y-axis (ranging from 0.2 to 1.0) against the 'Index of the largest eigenvalue' on the x-axis (ranging from 2 to 12). Multiple colored lines represent different years from 2011 to 2017, all showing a rapid increase in cumulative singular values. A vertical dashed red line is drawn at index 5. The right part is a heatmap titled 'PEMS-Stream' showing 'Years' on the y-axis (2011 to 2017) against 'Steps of saving models' on the x-axis (1 to 5). The color scale ranges from 0.76 (light green) to 0.86 (dark blue). The heatmap shows that the normalized cumulative singular values remain high (mostly above 0.8) across all years and steps.](55d2bfe1c3d04e86df8d7a104d802172_img.jpg)

Figure 4: Low-rank measurement. The figure consists of two parts. The left part is a line plot titled 'PEMS-Stream' showing 'Normalized Cumulative Singular Value' on the y-axis (ranging from 0.2 to 1.0) against the 'Index of the largest eigenvalue' on the x-axis (ranging from 2 to 12). Multiple colored lines represent different years from 2011 to 2017, all showing a rapid increase in cumulative singular values. A vertical dashed red line is drawn at index 5. The right part is a heatmap titled 'PEMS-Stream' showing 'Years' on the y-axis (2011 to 2017) against 'Steps of saving models' on the x-axis (1 to 5). The color scale ranges from 0.76 (light green) to 0.86 (dark blue). The heatmap shows that the normalized cumulative singular values remain high (mostly above 0.8) across all years and steps.

Figure 4: Low-rank measurement.

**Theoretical Analysis.** Below, we provide a theoretical analysis of the above empirical results.

**Proposition 2.** *Given the node prompt parameter matrix  $P \in \mathbb{R}^{n \times d}$ , there will always be two matrices  $A \in \mathbb{R}^{n \times k}$  and  $B \in \mathbb{R}^{k \times d}$  such that  $P$  can be approximated as  $AB$  when the nodes  $n$  grow large, and satisfy the following probability inequality:*

$$\Pr(\|P - AB\|_F \leq \epsilon \|P\|_F) \geq 1 - o(1) \text{ and } k = \mathcal{O}(\log(\min(n, d)))$$

where  $o(1)$  represents a term that becomes negligible even as  $n$  grows large.

*Proof.* For more details, refer to the supplementary materials in appendix A.2.  $\square$

**Tuning Principle II: Prompt Parameter Pool Can Continuously Satisfy the Low-rank Property.**

**Implementation Details.** Formally, we present the implementation details for compress process in continuous spatio-temporal forecasting scenarios based on the aforementioned analysis. Specifically, for the initial static stage, we approximate the original prompt parameter  $P^{(1)}$  using the product of the subspace parameter matrix  $A^{(1)}$  and the adjustment parameter matrix  $B$ . For subsequent periods  $\tau$ , we provide only the subspace parameter matrix  $A^{(\tau)}$  for the newly added node vectors, approximating the prompt parameter  $P^{(\tau)}$  through the product with the adjustment parameter matrix  $B$ . As analyzed, the dimensionality of the subspace parameter matrix  $A$  is significantly smaller than that of the prompt parameter  $P$ , while the number of parameters in the adjustment matrix  $B$  remains constant; thus, we effectively mitigate the inflation issue.

## 5 EXPERIMENTS

In this section, we conduct extensive experiments to investigate the following research questions:

- **RQ1:** Can EAC outperform previous methods in accuracy across various tasks? (*Effectiveness*)
- **RQ2:** Can EAC have a consistent improvement on various types of STGNNs? (*Universality*)
- **RQ3:** How efficient is EAC compared to different methods during the training phase? (*Efficiency*)
- **RQ4:** How many parameters does EAC require tuning compared to baselines? (*Lightweight*)
- **RQ5:** How does EAC compare to other common prompt-adaptive learning method? (*Simplicity*)

### 5.1 EXPERIMENTAL SETUP

**Dataset and Evaluation Protocol.** We use real-world spatio-temporal graph datasets from three domains: transportation, weather, and energy, encompassing common streaming spatio-temporal forecasting scenarios. The transportation dataset, *PEMS-Stream*, is derived from benchmark datasets in previous research (Chen et al., 2021b), covering dynamic traffic flow in Northern California from 2011 to 2017 across seven periods. The weather dataset, *Air-Stream*, originates from the real-time urban air quality platform of the Chinese Environmental Monitoring Center<sup>1</sup>, capturing dynamic air quality indicators at monitoring stations across various regions of China from 2016 to 2019 over

<sup>1</sup><https://air.cnemc.cn:18007/>

{6}------------------------------------------------

four periods. The energy dataset, *Energy-Stream*, comes from a spatial dynamic wind power forecasting dataset provided by a power company during the KDD Cup competition (Zhou et al., 2022), containing monitoring metrics for wind farms over a span of 245 days across four periods. During the experiments, each dataset’s temporal dimension is split into training, validation, and test sets in a 6:2:2 ratio for each period, employing an early stopping mechanism. According to the established protocol (Chen et al., 2021b), we use the past 12 steps to predict the next 3, 6, 12 steps, and the mean value. Evaluation metrics include *Mean Absolute Error (MAE)*, *Root Mean Square Error (RMSE)*, and *Mean Absolute Percentage Error (MAPE)* averaged over all periods. Further details regarding the datasets and evaluation metrics are available in Appendix C.1.

**Baseline and Parameter Setting.** Following the default settings (Chen et al., 2021b; Wang et al., 2023a; Lee & Park, 2024), we employ the same STGNN as the backbone network and consider the following baseline methods for comparison:

- **Pretrain-ST:** For each dataset, we train the STGNN backbone using only the spatio-temporal graph data from the first period and directly use this network to predict results on the test sets in subsequent periods.
- **Retrain-ST:** For each dataset, we train a new backbone network for the spatio-temporal graph data of each period and use the corresponding network to predict results on the test set of the current.
- **Online-ST:** For each dataset, we iteratively train a backbone network in an online manner, where the model weights from the previous period serve as initialization for the current period.
 - ‡ **Online-ST-AN:** Train on the complete node data of the current period’s spatio-temporal graph, tuning the entire model with the model trained from the previous year as initialization.
 - ‡ **Online-ST-NN:** Train on the newly added node data of the current period’s spatio-temporal graph, tuning the entire model with the model trained from the previous year as initialization. **TFMoE** (Lee & Park, 2024) improved this using a mixture of expert models technique.
 - ‡ **Online-ST-MN:** Train on the mixed node data (new nodes + some old nodes) of the current period’s spatio-temporal graph, tuning the entire model with the model trained from the previous year as initialization. Existing methods typically focus on this aspect, including *Traffic-Stream* (Chen et al., 2021b), *PECMP* (Wang et al., 2023b), and *STKEC* (Wang et al., 2023a).

For both the baseline method and our method, we set parameters uniformly according to the recommendations in previous paper (Chen et al., 2021b) to ensure fair comparison. Our only hyperparameter  $k$  is set to 6. We repeated each experiment five times and report the mean and standard deviation (indicated by gray  $\pm$ ) of all methods. More details about baseline settings, see Appendix C.2.

### 5.2 EFFECTIVENESS STUDY (RQ1)

**Overall Performance.** We report a comparison between EAC and typical schemes (including representative improved methods <sup>2</sup>) in Table 1, where the best results are highlighted in **bold pink** and the second-best results in underlined blue.  $\Delta$  indicates the reduction of MAE compared to the second-best result, or the increase of other results relative to the second best result. Moreover, due to the unavailability of official source code for *PECMP* and *TFMoE*, along with the more complex backbone network used by *TFMoE*, the comparisons may be unfair. Nonetheless, we also include comparable reported values, as shown in Table 2. Based on the results, we observe the following:

- ❶ **Pretrain-ST** methods generally yield the poorest results, especially on smaller datasets (i.e., *Engery-Stream*), aligning with the intuition that they directly use a pre-trained model for zero-shot forecasting in subsequent periods. Even with better pre-training on larger dataset (i.e., *Air-Stream*), performance remains mediocre.
- ❷ **Retrain-ST** methods also exhibit unsatisfactory results, as they rely on limited data to train specific phase models without effectively utilizing historical information gained from the pretrained model.
- ❸ **Online-ST-NN** methods perform poorly, as they fine-tune the pretrained model using only new node data differing from the old pattern. Despite *TFMoE*’s improvements through complex design, severe catastrophic forgetting remains an issue.
- ❹ **Online-ST-MN** methods strike a balance between performance and efficiency, showing some improvements, particularly on small datasets (e.g., *STKEC* in *Energy-Stream*), due to limited node pattern memory.
- ❺ **Online-ST-AN** methods typically achieve suboptimal results, as they fine-tune the pretrained model on the full data across different periods, approximating the performance boundary

<sup>2</sup>Notably, the core code of *STKEC* is not available, so we carefully reproduce and report average results.

{7}------------------------------------------------

Table 1: Comparison of the overall performance of the classical scheme and EAC .

| Datasets |  | <i>Air-Stream</i> (1087 → ... → 1202) |  |  |  | <i>PEMS-Stream</i> (655 → ... → 871) |  |  |  | <i>Energy-Stream</i> (103 → ... → 134) |  |  |  |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|
| Method | Metric | 3 | 6 | 12 | Avg. | 3 | 6 | 12 | Avg. | 3 | 6 | 12 | Avg. |
| <b>Retrain-ST</b> | MAE | 18.50 <sub>±0.20</sub> | 21.53 <sub>±0.20</sub> | 24.83 <sub>±0.20</sub> | 21.33 <sub>±0.20</sub> | 12.96 <sub>±0.14</sub> | 14.06 <sub>±0.10</sub> | 16.36 <sub>±0.11</sub> | 14.24 <sub>±0.12</sub> | 5.56 <sub>±0.14</sub> | 5.46 <sub>±0.12</sub> | 5.45 <sub>±0.09</sub> | 5.48 <sub>±0.12</sub> |
|  | RMSE | 20.20 <sub>±0.17</sub> | 34.31 <sub>±0.19</sub> | 39.61 <sub>±0.57</sub> | 33.77 <sub>±0.60</sub> | 20.88 <sub>±0.17</sub> | 22.96 <sub>±0.15</sub> | 26.95 <sub>±0.19</sub> | 23.20 <sub>±0.16</sub> | 5.75 <sub>±0.12</sub> | 5.70 <sub>±0.11</sub> | 5.80 <sub>±0.09</sub> | 5.72 <sub>±0.11</sub> |
|  | MAPE (%) | 22.72 <sub>±0.20</sub> | 27.67 <sub>±0.35</sub> | 32.50 <sub>±0.60</sub> | 27.54 <sub>±0.60</sub> | 18.51 <sub>±0.61</sub> | 19.98 <sub>±0.42</sub> | 23.31 <sub>±0.24</sub> | 20.30 <sub>±0.44</sub> | 54.35 <sub>±1.21</sub> | 54.61 <sub>±2.08</sub> | 55.60 <sub>±0.55</sub> | 54.74 <sub>±2.06</sub> |
|  | Δ | +1.58% | +1.03% | +0.52% | +0.99% | +1.25% | +1.00% | +1.17% | +1.13% | +4.70% | +2.24% | +0.73% | +2.23% |
| <b>Pretrain-ST</b> | MAE | 19.58 <sub>±0.20</sub> | 22.72 <sub>±0.16</sub> | 26.00 <sub>±0.20</sub> | 22.44 <sub>±0.18</sub> | 14.13 <sub>±0.28</sub> | 15.17 <sub>±0.26</sub> | 17.35 <sub>±0.29</sub> | 15.33 <sub>±0.27</sub> | 10.65 <sub>±0.00</sub> | 10.66 <sub>±0.02</sub> | 17.16 <sub>±0.42</sub> | 17.05 <sub>±0.39</sub> |
|  | RMSE | 31.46 <sub>±0.20</sub> | 36.78 <sub>±0.16</sub> | 41.96 <sub>±0.20</sub> | 36.15 <sub>±0.34</sub> | 21.77 <sub>±0.25</sub> | 23.79 <sub>±0.26</sub> | 27.73 <sub>±0.33</sub> | 24.04 <sub>±0.27</sub> | 10.88 <sub>±0.12</sub> | 10.92 <sub>±0.13</sub> | 11.02 <sub>±0.15</sub> | 10.93 <sub>±0.13</sub> |
|  | MAPE (%) | 24.05 <sub>±1.12</sub> | 28.46 <sub>±1.12</sub> | 33.48 <sub>±1.12</sub> | 28.16 <sub>±1.17</sub> | 30.86 <sub>±1.34</sub> | 32.07 <sub>±1.34</sub> | 34.45 <sub>±1.34</sub> | 32.20 <sub>±1.34</sub> | 171.88 <sub>±3.79</sub> | 172.77 <sub>±4.25</sub> | 174.07 <sub>±4.83</sub> | 172.71 <sub>±4.12</sub> |
|  | Δ | +6.99% | +6.61% | +5.26% | +6.25% | +10.39% | +8.97% | +7.29% | +8.87% | +100.56% | +99.62% | +216.08% | +218.09% |
| <b>Online-ST-AN</b> | MAE | 18.30 <sub>±0.20</sub> | 21.31 <sub>±0.20</sub> | 23.70 <sub>±0.49</sub> | 21.12 <sub>±0.51</sub> | 12.80 <sub>±0.06</sub> | 13.92 <sub>±0.05</sub> | 16.17 <sub>±0.10</sub> | 14.08 <sub>±0.05</sub> | 5.47 <sub>±0.08</sub> | 5.46 <sub>±0.09</sub> | 5.47 <sub>±1.11</sub> | 5.47 <sub>±0.08</sub> |
|  | RMSE | 28.54 <sub>±0.68</sub> | 33.87 <sub>±0.65</sub> | 39.33 <sub>±0.68</sub> | 33.28 <sub>±0.68</sub> | 20.66 <sub>±0.06</sub> | 22.73 <sub>±0.06</sub> | 26.64 <sub>±0.15</sub> | 22.96 <sub>±0.06</sub> | 5.62 <sub>±0.05</sub> | 5.66 <sub>±0.06</sub> | 5.76 <sub>±0.07</sub> | 5.67 <sub>±0.05</sub> |
|  | MAPE (%) | 21.43 <sub>±0.60</sub> | 27.45 <sub>±0.65</sub> | 32.38 <sub>±0.60</sub> | 27.24 <sub>±0.60</sub> | 17.86 <sub>±0.59</sub> | 19.37 <sub>±0.59</sub> | 22.92 <sub>±1.16</sub> | 19.73 <sub>±0.74</sub> | 52.70 <sub>±1.34</sub> | 53.25 <sub>±1.34</sub> | 54.50 <sub>±1.71</sub> | 53.36 <sub>±1.51</sub> |
|  | Δ | - | - | - | - | - | - | - | - | +3.01% | +2.24% | +1.10% | +2.05% |
| <b>Online-ST-NN</b> | MAE | 19.38 <sub>±1.12</sub> | 22.24 <sub>±1.12</sub> | 25.50 <sub>±1.12</sub> | 22.05 <sub>±1.12</sub> | 14.68 <sub>±0.60</sub> | 16.57 <sub>±1.25</sub> | 20.64 <sub>±2.25</sub> | 16.95 <sub>±1.51</sub> | 5.51 <sub>±0.05</sub> | 5.50 <sub>±0.05</sub> | 5.49 <sub>±0.07</sub> | 5.50 <sub>±0.05</sub> |
|  | RMSE | 29.57 <sub>±1.23</sub> | 34.40 <sub>±1.09</sub> | 39.62 <sub>±1.24</sub> | 33.97 <sub>±1.18</sub> | 24.30 <sub>±2.25</sub> | 28.21 <sub>±3.09</sub> | 36.77 <sub>±6.60</sub> | 29.05 <sub>±5.63</sub> | 5.65 <sub>±0.04</sub> | 5.68 <sub>±0.05</sub> | 5.76 <sub>±0.06</sub> | 5.70 <sub>±0.05</sub> |
|  | MAPE (%) | 23.99 <sub>±2.25</sub> | 28.02 <sub>±2.08</sub> | 33.17 <sub>±2.14</sub> | 27.96 <sub>±2.13</sub> | 18.93 <sub>±0.57</sub> | 20.69 <sub>±0.40</sub> | 24.89 <sub>±0.21</sub> | 21.15 <sub>±0.45</sub> | 54.98 <sub>±1.67</sub> | 55.10 <sub>±1.10</sub> | 55.62 <sub>±0.17</sub> | 55.17 <sub>±1.08</sub> |
|  | Δ | +5.90% | +4.36% | +3.23% | +4.40% | +14.68% | +19.03% | +27.64% | +20.38% | +3.76% | +2.99% | +1.47% | +2.61% |
| <b>TrafficStream</b> | MAE | 18.06 <sub>±1.21</sub> | 21.59 <sub>±0.06</sub> | 24.90 <sub>±0.76</sub> | 21.39 <sub>±1.02</sub> | 12.89 <sub>±0.06</sub> | 14.03 <sub>±1.11</sub> | 16.39 <sub>±2.29</sub> | 14.22 <sub>±1.17</sub> | 5.58 <sub>±0.03</sub> | 5.57 <sub>±0.05</sub> | 5.56 <sub>±0.07</sub> | 5.57 <sub>±0.05</sub> |
|  | RMSE | 29.06 <sub>±1.34</sub> | 34.22 <sub>±1.20</sub> | 39.54 <sub>±1.06</sub> | 33.65 <sub>±1.35</sub> | 20.78 <sub>±0.13</sub> | 22.90 <sub>±0.25</sub> | 26.98 <sub>±0.53</sub> | 23.16 <sub>±0.27</sub> | 5.73 <sub>±0.03</sub> | 5.76 <sub>±0.05</sub> | 5.86 <sub>±0.07</sub> | 5.77 <sub>±0.06</sub> |
|  | MAPE (%) | 24.23 <sub>±1.26</sub> | 28.12 <sub>±1.29</sub> | 32.99 <sub>±1.13</sub> | 28.03 <sub>±1.53</sub> | 17.86 <sub>±0.41</sub> | 19.50 <sub>±0.36</sub> | 23.43 <sub>±2.25</sub> | 19.95 <sub>±1.02</sub> | 53.87 <sub>±1.99</sub> | 54.06 <sub>±1.24</sub> | 54.92 <sub>±0.71</sub> | 54.16 <sub>±1.21</sub> |
|  | Δ | +1.96% | +1.13% | +0.80% | +1.27% | +0.70% | +0.79% | +1.36% | +0.99% | +5.08% | +4.30% | +3.14% | +3.91% |
| <b>STKEC</b> | MAE | 19.42 <sub>±1.29</sub> | 22.24 <sub>±1.12</sub> | 25.44 <sub>±1.06</sub> | 22.06 <sub>±1.20</sub> | 12.85 <sub>±0.06</sub> | 13.98 <sub>±0.06</sub> | 16.25 <sub>±0.06</sub> | 14.14 <sub>±0.06</sub> | 5.31 <sub>±0.27</sub> | 5.34 <sub>±0.25</sub> | 5.41 <sub>±0.15</sub> | 5.36 <sub>±0.22</sub> |
|  | RMSE | 30.28 <sub>±1.63</sub> | 35.09 <sub>±1.40</sub> | 40.11 <sub>±1.13</sub> | 34.61 <sub>±1.43</sub> | 20.73 <sub>±0.06</sub> | 22.81 <sub>±0.08</sub> | 26.73 <sub>±0.07</sub> | 23.04 <sub>±0.07</sub> | 5.50 <sub>±0.19</sub> | 5.56 <sub>±0.20</sub> | 5.72 <sub>±0.10</sub> | 5.59 <sub>±0.17</sub> |
|  | MAPE (%) | 25.21 <sub>±1.09</sub> | 28.83 <sub>±2.22</sub> | 33.30 <sub>±1.64</sub> | 28.71 <sub>±2.23</sub> | 17.87 <sub>±0.14</sub> | 19.25 <sub>±0.17</sub> | 22.33 <sub>±0.16</sub> | 19.53 <sub>±0.15</sub> | 47.53 <sub>±2.71</sub> | 50.93 <sub>±4.51</sub> | 52.10 <sub>±4.10</sub> | 51.04 <sub>±4.44</sub> |
|  | Δ | - | - | - | - | +0.39% | +0.43% | +0.49% | +0.42% | - | - | - | - |
| <b>EAC</b> | MAE | 18.11 <sub>±0.27</sub> | 20.87 <sub>±0.17</sub> | 24.15 <sub>±0.14</sub> | 20.75 <sub>±0.20</sub> | 12.65 <sub>±0.03</sub> | 13.45 <sub>±0.05</sub> | 14.92 <sub>±0.11</sub> | 13.53 <sub>±0.06</sub> | 5.08 <sub>±0.10</sub> | 5.09 <sub>±0.10</sub> | 5.15 <sub>±0.10</sub> | 5.10 <sub>±0.10</sub> |
|  | RMSE | 27.78 <sub>±0.47</sub> | 32.88 <sub>±0.42</sub> | 38.22 <sub>±0.31</sub> | 32.35 <sub>±0.40</sub> | 20.24 <sub>±0.06</sub> | 21.86 <sub>±0.09</sub> | 24.17 <sub>±0.17</sub> | 21.77 <sub>±0.10</sub> | 5.26 <sub>±0.10</sub> | 5.31 <sub>±0.10</sub> | 5.46 <sub>±0.09</sub> | 5.33 <sub>±0.10</sub> |
|  | MAPE (%) | 23.12 <sub>±0.20</sub> | 26.91 <sub>±0.07</sub> | 31.79 <sub>±0.08</sub> | 26.89 <sub>±0.12</sub> | 17.80 <sub>±0.06</sub> | 18.79 <sub>±0.06</sub> | 20.82 <sub>±0.16</sub> | 18.98 <sub>±0.08</sub> | 47.53 <sub>±2.71</sub> | 48.20 <sub>±2.68</sub> | 50.55 <sub>±2.60</sub> | 48.56 <sub>±2.67</sub> |
|  | Δ | -1.03% | -2.06% | -2.26% | -1.75% | -1.17% | -3.33% | -7.73% | -3.90% | -4.33% | -4.68% | -4.80% | -4.85% |

of continual-based methods while still suffering some knowledge forgetting. ❶ Our EAC consistently improves all metrics across all types of datasets. We attribute this to its ability to continuously adapt to the complex information and knowledge forgetting challenges inherent in continual spatio-temporal learning scenarios by tuning the prompt pool with heterogeneity-capturing parameters.

Table 2: Performance comparison of the improved method with EAC on *PEMS-Stream* benchmark.

| Model | Avg. @ MAE | Avg. @ RMSE | Simplicity | Lightweight |
|-|-|-|-|-|
| <i>TrafficStream</i> | 14.22 <sub>±0.13</sub> | 23.16 <sub>±0.27</sub> | ✗ | ✗ |
| <i>STKEC</i> | 14.14 <sub>±0.08</sub> | 23.04 <sub>±0.07</sub> | ✗ | ✗ |
| <i>PECMP</i> | 14.85 * | 24.62 * | ✗ | ✗ |
| <i>TFMoE</i> | 14.18 * | 23.54 * | ✓ | ✓ |
| EAC | 13.53 <sub>±0.06</sub> | 21.77 <sub>±0.10</sub> | ✓ | ✓ |

![Figure 5: Few-Shot Scenario Forecasting in PEMS-Stream benchmark. The figure contains two line graphs. The left graph shows 'Avg. @ RMSE' on the y-axis (ranging from 22 to 32) against 'Year' on the x-axis (2011 to 2017). The right graph shows '12 @ RMSE' on the y-axis (ranging from 20 to 32) against 'Year' on the x-axis (2011 to 2017). Both graphs compare five methods: Retrain-ST (blue circle), Pretrain-ST (orange triangle), Online-ST-AN (green diamond), EAC (red cross), and Online-ST-NN (cyan plus). In both graphs, EAC consistently shows the lowest RMSE values across all years, indicating superior performance in the few-shot forecasting scenario.](27b22513fc27a0ff5f230b062ad3112f_img.jpg)

Figure 5: Few-Shot Scenario Forecasting in PEMS-Stream benchmark. The figure contains two line graphs. The left graph shows 'Avg. @ RMSE' on the y-axis (ranging from 22 to 32) against 'Year' on the x-axis (2011 to 2017). The right graph shows '12 @ RMSE' on the y-axis (ranging from 20 to 32) against 'Year' on the x-axis (2011 to 2017). Both graphs compare five methods: Retrain-ST (blue circle), Pretrain-ST (orange triangle), Online-ST-AN (green diamond), EAC (red cross), and Online-ST-NN (cyan plus). In both graphs, EAC consistently shows the lowest RMSE values across all years, indicating superior performance in the few-shot forecasting scenario.

Figure 5: Few-Shot Scenario Forecasting in *PEMS-Stream* benchmark.

❶ (**Robustness**) All methods exhibit a decline in performance compared to the complete data scenario, with *Online-ST-NN* demonstrating significant sensitivity due to its inherent catastrophic forgetting issue. However, our EAC method exhibits controllable robustness across all years, outperforming all other methods. ❷ (**Adaptability**) The performance of all methods consistently declines as the periods extend, yet our EAC method demonstrates a relatively mild decline, particularly evident in the 12-step metrics, where it significantly outperforms all other approaches.

{8}------------------------------------------------

### 5.3 UNIVERSALITY STUDY (RQ2)

**Setting.** We further aim to demonstrate the universality of our EAC in enhancing the performance of various STGNN backbones, an aspect largely overlooked in previous studies. Specifically, STGNN can be categorized into spectral-based and spatial-based graph convolution operators, as well as recurrent-based, convolution-based, and attention-based sequence modeling operators. We select a representative operator from each category to form six distinct models, where the core architecture consists of two interleaved graph convolution modules and one sequence module. A detailed description of the different operators can be found in Appendix E. Additionally, we adapt different models to the prompt parameter pool proposed by EAC to compare the performance impact.

Table 3: Effect of EAC on the average performance of different STGNN component on the *PEMS-Stream*. C-based: Convolution-based, R-based: Recurrent-based, A-based: Attention-based.

| Methods | Spatial-based |  |  |  |  |  | Spectral-based |  |  |  |  |  |
|-|-|-|-|-|-|-|-|-|-|-|-|-|
|  | C-based |  | R-based |  | A-based |  | C-based |  | R-based |  | A-based |  |
| Metric | MAE | RMSE | MAE | RMSE | MAE | RMSE | MAE | RMSE | MAE | RMSE | MAE | RMSE |
| <i>w/o</i> | 14.07 <sub>±0.07</sub> | 22.93 <sub>±0.08</sub> | 13.23 <sub>±0.10</sub> | 21.36 <sub>±0.18</sub> | 14.69 <sub>±0.42</sub> | 23.33 <sub>±0.59</sub> | 14.01 <sub>±0.01</sub> | 22.76 <sub>±0.03</sub> | 13.73 <sub>±0.91</sub> | 21.87 <sub>±1.24</sub> | 14.64 <sub>±0.06</sub> | 23.12 <sub>±0.06</sub> |
| EAC | 13.72 <sub>±0.01</sub> | 22.14 <sub>±0.02</sub> | <b>12.83</b> <sub>±0.08</sub> | <b>20.59</b> <sub>±0.09</sub> | 14.64 <sub>±0.53</sub> | 22.79 <sub>±0.59</sub> | 13.62 <sub>±0.12</sub> | 21.80 <sub>±0.18</sub> | <b>13.00</b> <sub>±0.37</sub> | <b>20.80</b> <sub>±0.66</sub> | 13.69 <sub>±0.08</sub> | 21.65 <sub>±0.10</sub> |
| Δ | -2.48% | -3.44% | -3.02% | -3.60% | -0.34% | -2.31% | -2.78% | -4.21% | -4.66% | -4.89% | -6.48% | -6.35% |

**Result Analysis.** Due to page limitations, we present MAE and RMSE metrics at average time steps using the *PEMS-Stream* dataset. As shown in Table 3, we observe that:

- ➊ Our EAC consistently demonstrates performance improvements across different combinations, highlighting its universality for various architectures.
- ➋ Compared to spatial domain-based graph convolution operators, our EAC shows a more pronounced enhancement for spectral domain-based methods.
- ➌ The advantages of recurrent-based sequence modeling operators are particularly evident, achieving the best results, while attention-based methods perform the worst. This aligns with intuition, as directly introducing vanilla multi-head attention may lead to excessive parameters and over-fitting; however, our approach still provides certain gains in these cases.

### 5.4 EFFICIENCY & LIGHTWEIGHT STUDY (RQ3 & RQ4)

**Overall Analysis.** We first conduct a comprehensive comparison of the EAC with other baselines in terms of performance, training speed, and memory usage. All models are configured with the same batch size to ensure fairness. As illustrated in Figure 6, we visualize the performance, average tuning parameters, and average training time (per period) of different methods on both the smallest dataset (*Energy-Stream*) and the largest dataset (*Air-Stream*). Our observations are as follows:

![Figure 6: Efficiency & Lightwight & Performance Study. Two scatter plots comparing methods on Energy-Stream and Air-Stream datasets. The x-axis is Average Training Time (s / period) and the y-axis is MAE. Methods include EAC, Retrain-ST, Online-STAN, TrafficStream, STKEC, and Online-STNN. EAC consistently shows lower MAE and training time across both datasets.](9d8d3d909d7fdccb631c519df2b86e61_img.jpg)

Figure 6 consists of two scatter plots. The left plot is for the *Energy-Stream* dataset and the right plot is for the *Air-Stream* dataset. Both plots show MAE on the y-axis and Average Training Time (s / period) on the x-axis. The methods compared are EAC, Retrain-ST, Online-STAN, TrafficStream, STKEC, and Online-STNN. In both plots, EAC (red circle) is positioned at the bottom-left, indicating the lowest MAE and training time. Other methods are clustered at higher MAE and training time values. A dashed box labeled 'Memory Footprint' contains the values 1E3, 2E3, and 3E3, indicating the relative memory usage of the methods.

Figure 6: Efficiency & Lightwight & Performance Study. Two scatter plots comparing methods on Energy-Stream and Air-Stream datasets. The x-axis is Average Training Time (s / period) and the y-axis is MAE. Methods include EAC, Retrain-ST, Online-STAN, TrafficStream, STKEC, and Online-STNN. EAC consistently shows lower MAE and training time across both datasets.

Figure 6: Efficiency & Lightwight & Performance Study.

- ➊ On datasets with a smaller number of nodes, our EAC consistently outperforms the others, achieving superior performance with only half the number of tuning parameters. Furthermore, the average training time per period accelerates by a factor of 1.26 to 3.02. In contrast, other methods such as *TrafficStream* and *STKEC*, despite employing second-order subgraph sampling techniques, do not benefit from efficiency improvements due to the limited number of nodes, which typically necessitates covering the entire graph.
- ➋ On datasets with a larger number of nodes, although the EAC exhibits a slightly higher number of tuning parameters, the freezing of the backbone model

{9}------------------------------------------------

results in faster training speeds while still achieving superior performance. ❹ According to the tuning principle of compression, we set  $k = 2$  to replace 6, resulting in the EAC -*Efficient* version, which maintains relative performance superiority on larger datasets using only  $\sim 63\%$  parameters compared to others. This demonstrates the superiority of the compression principle we propose.

**Hyper-parameter Analysis.** We further examine our sole hyper-parameter  $k$ , proposed through the compress principle, and its mutual influence on performance and tuning parameters. As shown in Figure 7, the horizontal axis denotes the values of  $k$ , while the vertical axes depict the performance on the *PEMS-Stream*. The color of the bars indicates the averaged percentage of tuning parameters relative to the total number of parameters across all periods. Our observations are as follows:

![Figure 7: Hyper-parameter study in PEMS-Stream benchmark. The figure contains two bar charts. The left chart shows MAE (Mean Absolute Error) for k values 2, 4, 6, 8, 10, and 12. The right chart shows RMSE (Root Mean Square Error) for the same k values. Both charts include a dashed red line labeled '(Second Best)'. The bars are colored based on the 'Avg. Tuning Parm. Percentage (%)' as indicated by the color bar on the right, ranging from 20% (light green) to 100% (dark green). Annotations '59% Parm.' and '74% Parm.' are present in both charts.](f519a5be118c846f631c992412353fb9_img.jpg)

Figure 7: Hyper-parameter study in PEMS-Stream benchmark. The figure contains two bar charts. The left chart shows MAE (Mean Absolute Error) for k values 2, 4, 6, 8, 10, and 12. The right chart shows RMSE (Root Mean Square Error) for the same k values. Both charts include a dashed red line labeled '(Second Best)'. The bars are colored based on the 'Avg. Tuning Parm. Percentage (%)' as indicated by the color bar on the right, ranging from 20% (light green) to 100% (dark green). Annotations '59% Parm.' and '74% Parm.' are present in both charts.

Figure 7: Hyper-parameter study in *PEMS-Stream* benchmark.

❶ From right to left, as the value of  $k$  decreases, indicating a reduction in tuning parameters, the overall performance of the model deteriorates significantly, accompanied by increased volatility (*i.e.*, higher standard deviation). This aligns with our findings in Figure 4, as the effective representational information of the approximate prompt parameter pool is constrained with decreasing  $k$ , leading to a marked decline in performance. ❷ We set  $k = 6$  as a default. Although performance continues to improve with increasing  $k$ , the gains are minimal. Moreover, excessively high values of  $k$  may result in negative effects due to redundant parameters, leading to over-fitting. Consequently, our method achieves satisfactory performance using only approximately 59% of the tuning parameters, effectively balancing performance and parameter efficiency.

### 5.5 SIMPLICITY STUDY (RQ5)

**Simplicity Analysis.** Lastly, we aim to explore the simplicity of the node parameter prompt pool, as well as the effectiveness of the expansion and compression principle. We selected a common low-rank adaptation (*LoRA*) (Hu et al., 2021; Ruan et al., 2024) technique, which has

Table 4: Performance comparison of *LoRA-Based* method with EAC on *PEMS-Stream* benchmark.

| Model | 2 @ MAE 12 | @ RMSE 12 | @ MAPE | Avg. Time |
|-|-|-|-|-|
| <i>LoRA-Based</i> | 10.22 <sub>±0.13</sub> | 20.81 <sub>±0.25</sub> | 22.78 <sub>±1.03</sub> | 337.31 <sub>±23.90</sub> |
| EAC | 14.92 <sub>±0.11</sub> | 24.17 <sub>±0.17</sub> | 20.82 <sub>±0.18</sub> | 224.33 <sub>±26.35</sub> |

recently been widely used in large language models. Following the default architecture, we added low-rank adaptation layers to the sequence operators, setting the rank to 6, and fine-tuned the backbone model during each period. As shown in Table 4, we observe that simply applying *LoRA* layers without considering the specific spatio-temporal context of streaming parameters may not be highly effective. Moreover, our method enjoy shorter training times compared to *LoRA-based* approaches, further validating the superiority of our proposed expansion and compression tuning principle.

## 6 CONCLUSION

In this paper, we derive two fundamental tuning principle: expand and compress for continual spatio-temporal forecasting scenarios through empirical observation and theoretical analysis. Adhering to these principle, we propose a novel prompt-based continual forecasting method, EAC, which effectively adapts to the complexities of dynamic continual spatio-temporal forecasting problems. Experimental results across various datasets from different domains in the real world demonstrate that EAC possesses desirable characteristics such as simplicity, effectiveness, efficiency, and universality. In the future, we will further explore large-scale pre-training methods for spatio-temporal data, leveraging the tuning principle proposed in this paper, which could broadly benefit a wide range of downstream continual forecasting tasks.

 Rest of paper (reference and Appendix) is removed.