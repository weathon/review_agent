# A General Spatio-Temporal Backbone With Scalable Contextual Pattern Bank For Urban Continual Forecasting

Aoyu Liu, Yaying Zhang∗
Tongji University {liuaoyu, yaying.zhang}@tongji.edu.cn

## Abstract

With the rapid growth of spatio-temporal data fueled by IoT deployments and urban infrastructure expansion, accurate and efficient continual forecasting has become a critical challenge. Most existing Spatio-Temporal Graph Neural Networks rely on static graph structures and offline training, rendering them inadequate for realworld streaming scenarios characterized by graph expansion and distribution shifts. Although Continual Spatio-Temporal Forecasting methods have been proposed to tackle these issues, they often adopt backbones with limited modeling capacity and lack effective mechanisms to balance stability and adaptability. To overcome these limitations, we propose STBP, a novel framework that integrates a general spatio-temporal backbone with a scalable contextual pattern bank. The backbone extracts stable representations in the frequency domain and captures dynamic spatial correlations through lightweight linear graph attention. To support continual adaptation and mitigate catastrophic forgetting, the contextual pattern bank is updated incrementally via parameter expansion, enabling the capture of evolving node-level heterogeneity and relevance. During incremental training, the backbone remains fixed to preserve general knowledge, while the pattern bank adapts to new scenarios and distributions. Extensive experiments demonstrate that STBP
outperforms state-of-the-art baselines in both forecasting accuracy and scalability, validating its effectiveness for continual spatio-temporal forecasting. Code is available at https://github.com/Aoyu-Liu/STBP.

## 1 Introduction

With the rapid development of urban IoT sensing systems, spatio-temporal data such as traffic flow (Shao et al., 2022b) and air quality (Tian et al., 2025) observations continue to surge (Kumar et al., 2024; Hu et al., 2023; Fang et al., 2026). Conducting efficient and accurate forecasting on data streams has become a core task in the development of smart cities (Jin et al., 2024; Yang et al., 2025). Unlike traditional offline learning based on static assumptions, real-world urban environments are in a state of continuous evolution—dynamic changes in urban structure and behavioral patterns constantly drive the evolution of graph structures and data distributions. Spatio-Temporal Graph Neural Networks (STGNNs) (Kong et al., 2024; Gao et al., 2024; Liu & Zhang, 2025) have been widely used to model complex spatio-temporal dependencies. However, most existing models still adhere to the paradigm of "fixed topology + offline training": the graph structure is predefined and fixed during the training phase, and the model is deployed directly after training. Yet, as shown in Figure 1, this static assumption becomes difficult to sustain when the node set continuously expands or the connectivity dynamically reconstructs over time. If one relies solely on structural modifications and continuous fine-tuning to handle node increments, model performance often degrades significantly. Therefore, Continual Spatio-Temporal Forecasting (CSTF) (Miao et al.,
2024b; Chen & Liang, 2025; Ma et al., 2025b) has garnered increasing attention. Its goal is to achieve incremental learning and efficient inference on new data without repeatedly relying on retraining with historical data. As shown in Figure 1, typical CSTF approaches employ a general spatio-temporal
∗Corresponding author.

1

![1_image_0.png](1_image_0.png) 
backbone integrated with strategies such as regularization, replay, or dynamic architectures to adapt to graph structural expansion and mitigate catastrophic forgetting. However, two key issues in existing CSTF methods have not yet been adequately addressed. First, the general backbone adopted by most current methods is relatively simple (e.g., stacks of graph and temporal convolutions), making it difficult to effectively handle incremental scenarios characterized by dynamically changing spatio-temporal correlations and long-term distribution drift. Forcibly adapting existing STGNNs for continual learning often leads to performance degradation (Shao et al., 2024; Ma et al., 2025a). Second, continual optimization strategies based on dynamic structural expansion are often weakly coupled with the backbone—such as direct parameter expansion or prompt concatenation—making it challenging to achieve a good balance among model stability, adaptability, and interpretability. Based on the above issues, we argue that an ideal CSTF framework should simultaneously address the following four key challenges: ❶ handling distributional drift; ❷ modeling dynamic spatio-temporal correlations; ❸ *alleviating catastrophic forgetting;* and ❹ designing an incremental strategy that efficiently collaborates with the backbone. To this end, we bridge the gap between STGNNs and continual learning by introducing a generalpurpose spatio-temporal backbone with scalable contextual pattern bank (STBP). Specifically, the backbone in STBP leverages frequency-domain modules to extract stable spatio-temporal components, mitigating distributional drift. Simultaneously, a lightweight, scene-agnostic linear graph attention mechanism is introduced to model dynamic spatial correlations with low computational overhead. To mitigate catastrophic forgetting and support continuous graph structure expansion, we design a contextual pattern bank composed of trainable parameters. It incrementally updates knowledge via parameter expansion and interacts with the backbone through gating and attention mechanisms, thereby uncovering node relevance and heterogeneity, and gradually adapting to scenario expansion at low cost. Within this framework, the backbone is responsible for modeling general and stable patterns, while the contextual pattern bank captures node-related heterogeneous contexts, working collaboratively to adapt to continuously evolving environments.

Our main contributions are summarized as follows: ❶ We propose an efficient and general backbone tailored for continual forecasting tasks, capable of modeling dynamic spatial correlations and mitigating distribution shift; ❷ We design a prompt-based guidance mechanism using contextual pattern bank, supporting dynamic model adaptation and alleviating catastrophic forgetting; ❸ Extensive experiments on multiple real-world datasets demonstrate that STBP significantly outperforms state-of-the-art baselines in terms of forecasting accuracy, adaptability, and scalability.

Figure 1: Limitations of existing studies.

## 2 Related Work

Spatio-Temporal Forecasting. Early studies in spatio-temporal forecasting, including methods like STGCN (Yu et al., 2018) and DCRNN (Li et al., 2018), primarily focused on combining basic temporal and spatial elements for prediction tasks. These models typically depended on predefined geographic adjacency matrices, which limited their ability to capture the evolving nature of spatial correlations. In contrast, later advancements, such as GWNet (Wu et al., 2019), DGCRN (Li et al., 2023), and MegaCRN (Jiang et al., 2023b), addressed this limitation by incorporating adaptive adjacency matrices or learning spatial correlations directly from the data. This shift led to a notable improvement in forecasting accuracy. More recently, models like STID (Shao et al., 2022a), STAEformer (Liu et al., 2023a), and HimNet (Dong et al., 2024) have emphasized the significance of distinguishing spatial patterns to further enhance forecasting performance. These methods incorporate trainable components, including spatial embeddings, parameter pools, and contextual pattern bank, to more accurately capture spatial variations, boosting both prediction precision and model adaptability. Continual Spatio-Temporal Forecasting. TrafficStream (Chen et al., 2021), one of the pioneering frameworks in CSTF, integrates spatio-temporal modeling with continual learning by employing historical data replay and parameter smoothing to manage long-term streaming traffic data and achieve accurate traffic flow prediction. Building on this line of work, STKEC (Wang et al., 2023a) proposes an influence-based knowledge expansion strategy together with a memory-augmented knowledge consolidation mechanism, which better supports the scaling of transportation networks while alleviating catastrophic forgetting. PECPM (Wang et al., 2023b) leverages pattern matching to dynamically maintain a traffic pattern bank, enabling efficient, historical-data-free continual learning with improved accuracy. STRAP (Zhang et al., 2025) adopts retrieval-augmented learning, constructing multi-dimensional pattern libraries and using plug-and-play prompting to fuse retrieved patterns, thereby enhancing out-of-distribution (OOD) generalization and mitigating catastrophic forgetting. EAC (Chen & Liang, 2025) introduces prompt tuning via a dynamic prompt pool that expands and compresses over time, balancing adaptation to new nodes with knowledge preservation in a parameter-efficient manner. Additionally, UFCL (Miao et al., 2025) leverages federated learning to protect data privacy and employs a global replay buffer of synthetic spatio-temporal data, addressing the challenges of distributed streaming environments.

## 3 Preliminary

Definition 1 (Streaming Spatio-Temporal Graph). We define a streaming spatio-temporal graph as a sequence of evolving graphs G = {Gτ }
T
τ=1, where each graph Gτ = (Vτ , Eτ , Aτ ) represents the graph at incremental period τ . Here, Vτ denotes the node set, Eτ the edge set, and adjacency matrix Aτ ∈ R
Nτ ×Nτ connections between nodes. The number of nodes at period τ is denoted by Nτ = |Vτ |. The graph evolves incrementally as Gτ = Gτ−1 + ∆Gτ , where ∆Gτ captures structural or feature modifications between periods. Definition 2 (Continual Spatio-Temporal Forecasting). Continual spatio-temporal forecasting aims to develop an optimal predictive model at each stage based on dynamic, streaming spatio-temporal graph data. At each incremental period τ , given the current graph Gτ and historical observations Xτ ∈ R
Nτ ×Th , the goal is to predict future signals Yτ ∈ R
Nτ ×Tf as follows:

$\eqref{eq:walpha}$. 
Yˆτ = fθ(Gτ , Xτ ), (1)
where Th is the length of the historical observation window, and Tf is the forecasting horizon. The model fθ is parameterized by θ, and continually updated by minimizing:
θ
∗ τ 
= arg min θ E(Gτ ,Xτ ,Yτ )∼Dτ
[L(fθ(Gτ , Xτ ), Yτ )] , (2)
where L(·, ·) is a loss function, and Dτ denotes the data distribution at period τ .

## 4 Methodology 4.1 Overview Of Stbp

The workflow and architecture of STBP are shown in Figure 2. It consists of two core components: a general spatio-temporal backbone and a contextual pattern bank. The backbone, comprising temporal and spatial modules with a prediction layer, captures spatio-temporal correlations in evolving networks. The contextual pattern bank, made of trainable parameters, is dynamically expanded and fine-tuned as data evolves. While the backbone captures general, stable patterns, the contextual pattern bank adapts to environmental changes, focusing on context-specific patterns. Guided by prompts, both components collaborate to form an efficient and robust continual learning system. In terms of workflow, streaming spatio-temporal data is sequentially fed into the STBP. During the initial incremental training phase, the backbone and contextual pattern bank are jointly trained to capture spatio-temporal correlations from current data. In later stages, the backbone is frozen (denoted by a snowflake) to retain knowledge learned from historical data, while the contextual pattern bank is updated (denoted by a flame) through expansion and fine-tuning. These updates serve as prompts, guiding the frozen backbone to adapt to new data distributions. This continual learning process, driven by the interplay between backbone and contextual pattern bank, enables the model to progressively enhance its representation power and adaptability while preserving core functionality. For detailed workflow steps, refer to Algorithm 1 in Appendix A.3.2.

![3_image_0.png](3_image_0.png)

## 4.2 Contextual Pattern Bank

![3_image_1.png](3_image_1.png)

Recent studies (Shao et al., 2022a; Dong et al., 2024; Chen & Liang, 2025) have shown that incorporating node-specific trainable parameters into STGNNs can significantly enhance forecasting performance. Following this insight, we propose an expandable contextual pattern bank Pτ ∈ R
Nτ ×d, composed of trainable parameters, to consolidate historical spatio-temporal patterns and generalize to new ones, thereby mitigating catastrophic forgetting and continuously adapting to new incremental scenarios, where d denotes the feature dimension.

We posit that the model can utilize Pτ to effectively distinguish both the relevance and *heterogeneity* of nodes, enabling a more nuanced understanding of the underlying data structures. Here, *relevance* refers to shared behavioral patterns among nodes—such as similar trends or periodic fluctuations—while *heterogeneity* captures differences arising from distinct node functions or external factors such as geography, policy, or events.

To validate this hypothesis, we conduct a t-SNE-based analysis on Pτ trained on spatio-temporal datasets (see Figure 3), which reveals meaningful clustering patterns. Each cluster exhibits distinct characteristics, corresponding to *heterogeneity*, while nodes within the same cluster display similar temporal dynamics, reflecting *relevance*.

As shown in Figure 2, given a streaming spatio-temporal input Xτ ∈ R
Nτ ×Th , the backbone model Mθ, and contextual pattern bank Pτ ∈ R
Nτ ×d, the incremental learning process is formulated as:
Figure 3: Contextual pattern bank visualization.

$${\hat{\mathbf{Y}}}_{\tau}={\mathcal{M}}_{\theta}(\mathbf{X}_{\tau},\mathbf{P}_{\tau}).$$
Yˆτ = Mθ(Xτ , Pτ ). (3)
At the initial training stage (τ = 1), both the backbone and contextual pattern bank are jointly trained (denoted with flame). For subsequent stages (τ > 1), the backbone is frozen (denoted with snowflake), and only the contextual pattern bank is updated through expansion:

$$\mathbf{P}_{\tau}^{\prime}=\mathbf{P}_{\tau-1}\parallel\Delta\mathbf{P}_{\tau},$$
τ = Pτ−1 ∥ ∆Pτ , (4)
$\eqref{eq:walpha}$. 
where ∆Pτ ∈ R
(Nτ −Nτ−1)×drepresents newly introduced parameters for the current incremental period. Only the expanded contextual pattern bank P′τ ∈ R
Nτ ×dis fine-tuned during training.

Notably, even without explicit clustering constraints, the contextual pattern bank autonomously distinguishes heterogeneous and relevant nodes through data-driven parameter learning and promptbased interactions with the backbone, driven by the prediction task. This strategy ensures that the backbone retains previously acquired knowledge, while the contextual pattern bank continually adapts to evolving distributions. It incrementally expands to represent an increasingly diverse set of environmental patterns, thereby avoiding the inadequacy exhibited by fixed models in novel scenarios. Distinct from existing work (Wang et al., 2023a; Chen & Liang, 2025; Wang et al., 2023b), we introduce a *Prompt-Based Guidance* (Peebles & Xie, 2023; Zhang et al., 2023) mechanism to enhance Pτ 's capacity to model both node-level relevance and heterogeneity. Specifically, the contextual pattern bank comprises three groups of trainable parameters: P
(i)
τ ∈ R
Nτ ×dfor i ∈ 0, 1, 2. As illustrated in Figure 2, these components interact with the backbone's hidden representation Hτ via the following prompt-based gating function:

$$\mathbf{H}_{\tau}^{\prime}=\mathbf{P}_{\tau}^{(1)}\cdot h_{\theta}(\mathbf{H}_{\tau}\cdot(1+\mathbf{P}_{\tau}^{(0)})),$$
$$({\boldsymbol{5}})$$
τ)), (5)
where hθ denotes an arbitrary submodule within the backbone. This gating mechanism enables adaptive modeling of node heterogeneity. Additionally, P
(2)
τ acts as a key embedding in the attention module, guiding the backbone to generalize correlation-aware information under task constraints. Importantly, since the contextual pattern bank encodes high-level abstractions rather than raw historical data, our method supports knowledge retention without revisiting prior data—offering advantages in privacy protection and *storage efficiency*.

## 4.3 General Spatio-Temporal Backbone

While the contextual pattern bank mitigates catastrophic forgetting in continual learning, it lacks the ability to model dynamic spatio-temporal correlations and handle distributional drift. To address this, we design a **general spatio-temporal backbone** aimed at handling distributional drift, spatiotemporal correlation modeling, and graph scalability during continual learning. The term *general* implies that the backbone is independent of the number of nodes and does not rely on any predefined adjacency matrix, making it adaptable to arbitrary spatio-temporal data structures. As shown in Figure 2, the backbone operates as follows: the input spatio-temporal data first passes through a **frequency-domain network** (FreNet), which maps it into high-dimensional temporal representations and extracts stable components via frequency domain analysis. A **dual-stream**
linear graph attention (DLGA) module then captures dynamic spatial correlations, followed by a feedforward layer with a multilayer perceptron for enhanced nonlinear expressivity. Finally, the features are reconstructed to their original shape by another FreNet and passed through a prediction layer. We detail the FreNet and DLGA modules below.

Frequency-Domain Network. Spatio-temporal data in evolving environments often suffer from distributional drift (Wang et al., 2024; Ji et al., 2025; Zhou et al., 2023). Although the contextual pattern bank helps retain stable knowledge, we further address this issue through a dedicated frequency-domain analysis (Xia et al., 2023). FreNet is designed to capture temporal correlations while emphasizing stable components in the data, such as periodicity and trends, which are more resilient to distributional changes (Liu & Zhang, 2025). Specifically, STBP employs two FreNets—one at the beginning and one at the end of the backbone (Figure 2). The first maps input data Xτ ∈ R
Nτ ×Th through a linear layer into a high-dimensional representation Hτ ∈ R
Nτ ×d, which is then transformed to the frequency domain using a Fast Fourier Transform (FFT). A learnable frequency-domain embedding Fτ ∈ C
(
d 2 
+1) adaptively highlights stable features. This process is formalized as:
Hfτ = IFFT(FFT(Hτ ) ⊙ Fτ ), (6)
where Hfτ ∈ R
Nτ ×dis further processed by a linear layer. The resulting representation Hfτthen interacts with the contextual pattern bank component P
(0)
τ via gating-based prompt guidance (Eq.

5) to produce Hs τ ∈ R
Nτ ×d, which serves as input to the subsequent DLGA module. The second FreNet performs an inverse operation, restoring the feature shape to R
Nτ ×Th . Compared to traditional temporal modules like RNNs (Li et al., 2018; Bai et al., 2020) or TCNs (Zheng et al., 2023; Fang et al., 2023), FreNet offers higher computational efficiency and enhanced ability to extract

stable low-frequency components (e.g., periodicity and trends) while suppressing high-frequency noise, thereby obtaining more robust temporal representations that are resilient to distributional drift across periods and scenarios. Dual-Stream Linear Graph Attention. After obtaining stable components, it remains essential to capture complex spatial interactions and time-varying node correlations. An effective spatial module must adaptively learn node correlations in a data-driven manner, maintain computational efficiency, and scale to growing graphs. Graph attention mechanisms (Velickovi ˇ c et al., 2018) have ´ emerged as promising solutions, enabling dynamic correlation modeling without relying on fixed adjacency matrices. However, conventional graph attention (Zheng et al., 2020; Jiang et al., 2023a;
Liu et al., 2023a) incurs O(N2) complexity, limiting its scalability. To overcome this, we propose
DLGA (Figure 2), which improves efficiency using a *random feature mapping*-based linear attention mechanism (Katharopoulos et al., 2020). Moreover, DLGA introduces a **dual-stream structure** by
incorporating the contextual pattern bank P
(2)
τ ∈ R
Nτ ×das an additional key. This enables the model
to assess the relationship between evolving input patterns and stored knowledge. Formally:
$${\bf Q}={\bf W}_{q}{\bf H}_{\tau}^{s},\quad{\bf K}={\bf W}_{k}{\bf H}_{\tau}^{s},\quad{\bf V}={\bf W}_{v}{\bf H}_{\tau}^{s},$$
, (7)
$$\mathbf{H}_{\tau}^{s^{\prime}}=\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{P}_{\tau}^{(2)})$$ $$=\text{Softmax}(\mathbf{Q}\mathbf{K}^{\top}+\mathbf{Q}(\mathbf{P}_{\tau}^{(2)})^{\top})\mathbf{V},$$ $$\text{Attention}(\mathbf{Q},\mathbf{K},\mathbf{V},\mathbf{P}_{\tau}^{(2)})\approx(\phi(\mathbf{Q})\phi(\mathbf{K})^{\top}+\phi(\mathbf{Q})\phi(\mathbf{P}_{\tau}^{(2)})^{\top})\mathbf{V}$$ $$=\phi(\mathbf{Q})\left(\phi(\mathbf{K})^{\top}\mathbf{V}+\phi(\mathbf{P}_{\tau}^{(2)})^{\top}\mathbf{V}\right).$$
$$({\mathfrak{s}})$$
$$\left(7\right)$$

$$(9)$$

.(9)
Here, Wq, Wk, and Wv are trainable projection matrices. Hsτand Hs
′
τ 
∈ R
Nτ ×d denote the input and the spatially enriched representation passed to the feedforward layer of the DLGA module, respectively. The function ϕ(·) denotes a random feature mapping, with Softmax used for approximation in our implementation. For further details on the approximation derivation, see Appendix A.3.1. Notably, the linear attention approximation does not explicitly construct an adjacency matrix. Instead, it implicitly models dynamic correlations by reordering operations in the attention computation. DLGA reduces computational complexity from quadratic to linear, while preserving dynamic spatial modeling and seamlessly integrating prompt-based knowledge from the contextual pattern bank.

## 5 Experiment

5.1 EXPERIMENTAL SETTINGS Datasets. We evaluate our model on three real-world streaming spatio-temporal datasets from the traffic and meteorology domains. The traffic datasets, **PEMS-Stream** (Chen et al., 2001) and CA-Stream (Liu et al., 2023b), consist of traffic flow measurements provided by the California Department of Transportation (CalTrans), with a sampling interval of 5 minutes. The meteorological dataset, **AIR-Stream** (Chen & Liang, 2025), is derived from urban air quality platform of the Chinese Environmental Monitoring Center, with hourly sampling intervals. To ensure fair evaluation, all datasets are split into training, validation, and test sets using a fixed ratio of 6:2:2. For each prediction task, the model is trained to forecast the next 12 time steps based on the previous 12 observations. Detailed dataset statistics are provided in Appendix A.4.1.

Baselines and Metrics. We select representative models from two categories as baselines: ▷ Conventional spatio-temporal forecasting models, including lightweight spatiotemporal architectures such as **GWNet** (Wu et al., 2019), **STID** (Shao et al., 2022a), and iTransformer (Liu et al., 2024b). These models are adapted specifically for incremental training in our experiments. ▷ Continual spatio-temporal forecasting models, including TrafficStream, **STKEC** (Wang et al., 2023a), **PECPM** (Wang et al., 2023b), **STRAP** (Zhang et al., 2025), and EAC (Chen & Liang, 2025). The performance of all models is evaluated using the following metrics: Mean Absolute Error (MAE), Root Mean Squared Error (**RMSE**), and Mean Absolute Percentage Error (**MAPE**). More details on this are included in Appendix A.4.2.

## 5.2 Main Results

The main experimental results are summarized in Table 1, which reports the metrics averaged over all incremental periods. We also present the results at specific forecasting horizons (3, 6, and 12 time steps ahead), together with the overall average across horizons. STGNNs, including GWNet and STID,

| Table 1: Main experimental results. Bold: best, underline: second best.   |                                                                 |                                                                 |                                                                 |       |     |      |
|---------------------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|-----------------------------------------------------------------|-------|-----|------|
| Dataset                                                                   | MetricHorizon GWNet                                             | STID                                                            | iTransformerTrafficStream STKEC PECPM                           | STRAP | EAC | STBP |
| 3                                                                         | 19.64±0.1224.34±0.13 17.63±0.76                                 | 14.23±0.09 14.29±0.1214.26±0.1314.30±0.11 13.86±0.16 11.62±0.09 |                                                                 |       |     |      |
| MAE                                                                       | 6                                                               | 19.68±0.1925.45±0.21 20.82±0.76                                 | 16.43±0.03 16.44±0.1116.35±0.1216.34±0.10 15.40±0.19 12.26±0.10 |       |     |      |
| 12                                                                        | 20.63±0.0929.42±0.38 28.33±0.86                                 | 21.76±0.07 21.66±0.1121.46±0.1921.52±0.15 18.90±0.28 13.47±0.08 |                                                                 |       |     |      |
| Avg. 19.87±0.1026.07±0.23 21.60±0.79                                      | 16.95±0.03 16.96±0.0916.86±0.1216.88±0.10 15.67±0.20 12.31±0.07 |                                                                 |                                                                 |       |     |      |
| PEMS-Stream                                                               | 3                                                               | 32.20±0.1739.37±0.13 28.20±1.15                                 | 23.00±0.09 23.08±0.1423.07±0.1523.06±0.13 22.26±0.23 19.20±0.13 |       |     |      |
| 6                                                                         | 32.34±0.3240.86±0.19 33.80±1.13                                 | 26.87±0.04 26.93±0.1526.76±0.2026.71±0.14 24.99±0.28 20.51±0.15 |                                                                 |       |     |      |
| RMSE                                                                      | 12                                                              | 33.73±0.0946.20±0.43 45.98±1.25                                 | 35.29±0.11 35.19±0.1134.77±0.3734.80±0.19 30.56±0.45 22.67±0.13 |       |     |      |
| Avg. 32.59±0.1841.67±0.21 34.88±1.17                                      | 27.52±0.05 27.56±0.1127.37±0.2027.35±0.13 25.30±0.29 20.52±0.11 |                                                                 |                                                                 |       |     |      |
| 3                                                                         | 27.47±0.6937.79±2.23 32.46±3.04                                 | 18.34±0.67 18.54±0.6118.19±0.6618.69±0.52 18.35±0.31 15.00±0.24 |                                                                 |       |     |      |
| MAPE                                                                      | 6                                                               | 27.22±0.5839.70±2.43 36.73±3.84                                 | 20.77±0.71 20.64±0.4820.79±0.5721.33±0.41 20.11±0.36 15.55±0.26 |       |     |      |
| (%)                                                                       | 12                                                              | 29.38±1.1847.94±2.91 54.31±4.66                                 | 27.88±0.26 27.05±0.6228.33±0.5228.20±1.10 24.30±0.57 16.75±0.23 |       |     |      |
| Avg. 27.79±0.7641.09±2.49 39.63±3.81                                      | 21.66±0.54 21.50±0.5221.73±0.4522.17±0.46 20.42±0.41 15.65±0.21 |                                                                 |                                                                 |       |     |      |
| 3                                                                         | 23.49±0.8027.71±0.23 20.16±0.06                                 | 17.82±0.26 17.69±0.1917.93±0.1223.59±0.61 17.66±0.37 15.01±0.18 |                                                                 |       |     |      |
| 6                                                                         | 23.31±0.6928.93±0.26 24.37±0.06                                 | 20.38±0.17 20.41±0.0420.33±0.0925.38±0.68 19.68±0.54 15.78±0.07 |                                                                 |       |     |      |
| MAE                                                                       | 12                                                              | 24.78±0.8633.61±0.45 34.05±0.06                                 | 26.92±0.53 27.05±0.1726.68±0.1931.10±0.89 24.86±1.33 17.19±0.09 |       |     |      |
| Avg. 23.73±0.7529.71±0.28 25.34±0.05                                      | 21.09±0.29 21.09±0.1321.04±0.1126.25±0.62 20.20±0.69 15.77±0.09 |                                                                 |                                                                 |       |     |      |
| CA-Stream                                                                 | 3                                                               | 35.87±0.9841.53±0.31 31.58±0.09                                 | 28.01±0.22 28.02±0.1928.00±0.1634.73±0.74 27.46±0.46 24.37±0.27 |       |     |      |
| 6                                                                         | 35.68±0.8843.14±0.35 37.76±0.10                                 | 32.19±0.22 32.43±0.0531.94±0.0937.97±0.86 30.64±0.83 25.71±0.22 |                                                                 |       |     |      |
| RMSE                                                                      | 12                                                              | 37.57±1.1149.18±0.58 51.24±0.10                                 | 41.59±0.64 42.08±0.2141.14±0.3046.74±1.36 37.77±1.94 28.08±0.14 |       |     |      |
| Avg. 36.20±0.9644.12±0.37 38.94±0.09                                      | 33.01±0.35 33.24±0.1332.77±0.1739.05±0.80 31.18±0.99 25.70±0.16 |                                                                 |                                                                 |       |     |      |
| 3                                                                         | 24.61±0.9529.24±0.65 21.76±0.17                                 | 17.05±0.41 16.60±0.1917.63±0.9119.11±0.49 18.26±1.88 14.22±0.03 |                                                                 |       |     |      |
| MAPE                                                                      | 6                                                               | 24.44±0.8030.66±0.78 26.76±0.22                                 | 19.22±0.30 18.98±0.1719.74±0.9220.48±0.39 19.45±1.16 14.85±0.07 |       |     |      |
| (%)                                                                       | 12                                                              | 25.71±0.8136.88±1.29 39.81±0.38                                 | 25.47±0.46 24.99±0.2925.94±1.0924.97±0.59 24.52±1.10 16.20±0.08 |       |     |      |
| Avg. 24.79±0.8531.73±0.86 28.34±0.20                                      | 19.98±0.30 19.61±0.1920.49±0.9121.15±0.47 20.17±1.25 14.94±0.05 |                                                                 |                                                                 |       |     |      |
| 3                                                                         | 28.48±1.4332.85±0.21 22.37±0.76                                 | 20.73±0.40 20.95±0.1720.82±0.3521.41±0.33 20.41±0.36 20.00±0.14 |                                                                 |       |     |      |
| 6                                                                         | 29.79±0.8933.15±0.22 26.22±0.48                                 | 25.64±0.34 25.54±0.0825.54±0.1926.12±0.34 25.20±0.29 24.70±0.30 |                                                                 |       |     |      |
| MAE                                                                       | 12                                                              | 31.30±0.5233.88±0.25 29.45±0.31                                 | 29.04±0.23 28.94±0.1228.95±0.1129.38±0.31 28.57±0.42 28.28±0.63 |       |     |      |
| Avg. 29.66±1.0133.23±0.22 25.53±0.56                                      | 24.58±0.34 24.63±0.1124.60±0.2125.16±0.32 24.21±0.43 23.64±0.23 |                                                                 |                                                                 |       |     |      |
| AIR-Stream                                                                | 3                                                               | 44.38±2.0451.24±0.28 34.98±1.18                                 | 32.80±0.57 33.13±0.2833.07±0.5233.72±0.41 32.19±0.57 32.15±0.24 |       |     |      |
| RMSE                                                                      | 6                                                               | 46.22±1.2851.61±0.31 40.95±0.73                                 | 40.41±0.53 40.38±0.2040.48±0.3941.13±0.4039.63±0.43 39.81±0.26  |       |     |      |
| 12                                                                        | 48.34±0.8552.55±0.39 45.70±0.55                                 | 45.54±0.47 45.53±0.2745.63±0.2746.07±0.3444.65±0.63 44.97±0.97  |                                                                 |       |     |      |
| Avg. 46.01±1.4651.72±0.33 39.67±0.91                                      | 38.58±0.53 38.70±0.2638.76±0.4139.37±0.38 37.83±0.60 37.76±0.30 |                                                                 |                                                                 |       |     |      |
| 3                                                                         | 38.02±2.6043.52±0.64 28.64±1.28                                 | 26.33±0.30 26.24±0.3025.79±0.5026.80±0.36 26.06±0.71 24.64±0.16 |                                                                 |       |     |      |
| MAPE                                                                      | 6                                                               | 39.98±1.7044.12±0.57 34.91±0.68                                 | 33.33±0.21 33.10±0.2832.97±0.1833.30±0.19 32.88±0.64 30.66±0.42 |       |     |      |
| (%)                                                                       | 12                                                              | 42.37±1.1445.06±0.62 40.79±0.39                                 | 39.27±0.24 39.02±0.1838.67±0.0238.87±0.24 38.85±0.67 36.23±0.52 |       |     |      |
| Avg. 39.87±1.8744.16±0.60 34.15±0.76                                      | 32.29±0.29 32.12±0.2131.82±0.1932.37±0.28 31.77±0.53 29.70±0.35 |                                                                 |                                                                 |       |     |      |

rely on static graph assumptions and are not designed for continual learning. Following prior work (Chen & Liang, 2025), we therefore retrain the backbone from scratch at each incremental stage using only data from the current period. In contrast, iTransformer is scenarioagnostic, so we adopt an **online** training regime: at each stage it is trained on the complete node set of the current spatio-temporal graph, initialized from the previous period's weights, enabling end-to-end fine-tuning. More detailed experimental results are provided in Appendix A.4.4. Results of conventional methods. As shown in Table 1, STGNNs trained from scratch achieve only poor performance on all datasets. Although these methods work well under static assumptions, they fail to exploit past spatio-temporal knowledge, resulting in unsatisfactory performance. In contrast, iTransformer performs better by leveraging historical spatio-temporal information through online training, but it still suffers from catastrophic forgetting and is therefore not an ideal solution.

Results of CSTF methods. The best-performing models are those that explicitly mitigate catastrophic forgetting, including CSTF methods such as PECPM, STRAP, and EAC. Compared with full-parameter fine-tuning strategies (e.g., PECPM, STKEC, TrafficStream), lightweight promptbased adaptation on a frozen backbone (e.g., EAC, STRAP, STBP) yields higher average accuracy, highlighting the benefits of dynamically tuning only a small set of parameters. Nevertheless, STRAP performs notably poorly on CA-Stream, indicating that retrieval-based pattern matching struggles Table 2: Comparison of few-shot forecasting performance.

Model PEMS-Stream 10% **CA-Stream 10%**
MAE RMSE MAPE (%) **MAE RMSE MAPE (%)**
GWNet 30.15±1.06 45.30±1.59 48.80±3.85 33.73±0.89 50.80±1.43 36.52±0.86 STID 33.42±2.90 50.63±3.73 63.96±12.60 37.09±0.52 55.10±0.69 39.18±1.12 iTransformer 20.99±0.19 32.67±0.25 49.11±1.62 25.43±0.08 39.01±0.10 28.39±0.54 TrafficStream 17.23±0.08 27.49±0.17 27.63±0.43 21.28±0.19 33.25±0.22 20.45±0.45 STKEC 17.75±0.12 28.23±0.13 27.80±0.88 21.20±0.13 33.20±0.08 20.23±0.46 PECPM 17.05±0.02 27.20±0.07 29.08±1.90 21.48±0.15 33.33±0.13 21.25±0.86 STRAP 17.68±0.10 27.98±0.14 31.67±2.88 26.34±0.79 39.39±1.09 21.34±0.45 EAC 16.13±0.05 25.57±0.06 24.02±1.23 20.94±0.70 32.15±1.00 21.37±1.53 STBP 13.58±0.0522.24±0.13 17.89±0.29 17.11±0.0327.48±0.16 17.60±0.30

![7_image_0.png](7_image_0.png)

![7_image_1.png](7_image_1.png)

in extreme incremental scenarios with rapid, large-scale topology expansion. Overall, our proposed STBP outperforms all competing models. Compared with the best baseline, STBP reduces the average MAE by 21.44%, **21.93%**, and **2.35%** on the PEMS-Stream, CA-Stream, and AIR-Stream datasets, respectively. This gain stems from the bridge it establishes between STGNNs and CSTF methods: the carefully designed general spatio-temporal backbone and contextual pattern bank jointly capture dynamic spatio-temporal correlations, thereby mitigating catastrophic forgetting and alleviating distributional drift. Results of few-shot forecasting task. To further evaluate the robustness of the proposed model under low-resource scenarios, we construct a few-shot training setting and compare it against existing baselines. Specifically, we simulate a few-shot setting in which the sample size of the first incremental period is kept unchanged, while the training set size for subsequent periods is reduced to only 10%
of the original. The test set size remains fixed throughout. As shown in Table 2, STBP consistently outperforms all other methods, highlighting its strong ability to extract meaningful patterns from limited data. CSTF baselines are more resilient to low-resource conditions than conventional STGNNs
(e.g., GWNet, STID). This demonstrates that when data is extremely scarce, conventional models struggle to capture stable spatio-temporal patterns, whereas CSTF methods can leverage knowledge accumulated from historical stages to adapt more quickly to new nodes. The continual learning mechanism effectively mitigates catastrophic forgetting, allowing the model to continuously utilize previously learned general features during incremental learning.

## 5.3 Ablation Study & Parameter Sensitivity Analysis

Ablation Study Settings. To validate the core contributions of STBP, we design the following variants for ablation experiments: ❶ **Retrain**: The contextual pattern bank is removed. Similar to GWNet and STID, a new backbone is trained for each incremental period using the spatio-temporal graph data of that period, with the corresponding model predicting the results for the current test set. ❷ **Online**: The contextual pattern bank is removed. Similar to iTransformer, the model is trained on the complete node data of the current spatio-temporal graph and initialized with the model from the previous period, allowing for adjustments across the entire model. ❸ **w/o Backbone**: The contextual pattern bank is retained, but the spatio-temporal backbone is replaced with the ones used in TrafficStream, STKEC, and EAC—i.e., replacing FreNet and DLGA with CNN and GCN. ❹ w/o DLGA: The DLGA module in the spatio-temporal backbone is ablated. ❺ EAC: We also include EAC,
which follows a similar approach, for comparison in the ablation study. Ablation findings. As shown in Figure 4, the ablation results demonstrate that parameter expansion in the contextual pattern bank, together with spatio-temporal pattern distinction and prompt guidance, is essential for alleviating catastrophic forgetting in continual learning. The performance of the Retrain and **Online** variants supports this conclusion. Notably, even without the contextual pattern

![8_image_0.png](8_image_0.png)

![8_image_1.png](8_image_1.png)

bank on traffic datasets, the spatio-temporal backbone alone attains performance comparable to EAC under online training, highlighting the critical role of real-time dynamic correlation modeling and temporal distribution-drift mitigation in adapting to new incremental tasks. The performance drop observed in the **w/o Backbone** variant further confirms the indispensability of the general backbone and highlights the portability and adaptability of the pattern bank across different backbone architectures. Moreover, removing the DLGA module leads to significant performance degradation, validating its role in capturing dynamic spatial correlations and integrating prompt-based knowledge.

The FreNet module also makes a notable contribution by improving computational efficiency and enhancing the extraction of stable temporal components.

Parameter Sensitivity Analysis. Additionally, we perform a sensitivity analysis on the adjustable hyperparameter d in STBP. In STBP, d represents the feature dimension for each module's feature mapping, as well as the feature dimension of parameters in the contextual pattern bank. The analysis results are shown in Figure 5. Increasing d enhances the model's overall parameter count and improves its expressive power. However, the performance gains from increasing d do not grow indefinitely; after reaching a certain threshold, the performance gain stabilizes. Further increases in d not only fail to improve performance but may also lead to negative effects, causing parameter redundancy. More parameter sensitivity analysis can be found in Appendix A.4.5.

## 5.4 Case Study

To illustrate the distinction and expandability of the contextual pattern bank in STBP, we apply t-SNE to reduce the dimensionality of Pτ ∈ R
Nτ ×d on the PEMS-Stream dataset. As shown in Figure 6, each point represents a graph node. Initially untrained, the pattern bank shows a chaotic distribution. After incremental training, clear clusters emerge. Nodes within the same cluster exhibit similar periodic and trend patterns in their traffic data, while those in different clusters (e.g., Clusters 1–3) show distinct behaviors. New nodes from later stages (e.g., Nodes 693, 809, 834 in 2017) are

![9_image_0.png](9_image_0.png)

correctly grouped into existing clusters, demonstrating that the pattern bank effectively distinguishes and generalizes spatio-temporal patterns through parameter fine-tuning, enabling continual adaptation.

In addition, we conduct an intuitive comparison of the forecasting performance between STBP and EAC in real-world application scenarios. As shown in Figure 7, we select representative nodes from three datasets for visualization. Compared to EAC, STBP more accurately captures dynamic trends, and its predictions demonstrate higher practical relevance in real-world continual learning environments. Additional case studies on other datasets can be found in Appendix A.4.6.

## 5.5 Efficiency Study

An effective CSTF method must balance scalability, computational cost, and performance. We evaluate the efficiency of STBP against baselines under the same settings. As shown in Figure 8, the average computational cost per period on PEMS-Stream and AIR-Stream is reported, with scatter size indicating GPU memory usage. We further analyze the impact of linear attention, full attention, and removal of the contextual pattern bank using a toy dataset. Results indicate that non-continual methods—such as GWNet, STID, and iTransformer—require global parameter adjustments at each phase, impairing efficiency. iTransformer, in particular, incurs high memory overhead due to quadratic attention complexity. Even lightweight non-continual models exhibit limited efficiency in incremental training.

In contrast, CSTF methods such as EAC, TrafficStream, and STKEC achieve higher efficiency through lightweight backbones and localized parameter tuning. While PECPM and STRAP maintain low memory usage, their training speeds remain modest. Despite its more complex backbone, STBP incurs only minimal overhead compared to models like EAC, thanks to optimizations including frequencydomain processing and linear attention. This enables STBP to deliver substantial performance gains with negligible cost increase. Results on the toy dataset confirm that linear attention reduces computational load effectively. As node count grows, the contextual pattern bank introduces only linear additional cost through its lightweight interaction with the backbone, avoiding exponential overhead. Furthermore, on CA-Stream, STBP maintains state-of-the-art performance even under drastic graph expansion, demonstrating strong scalability.

## 6 Conclusion

In this work, we propose STBP, a novel framework for continual spatio-temporal forecasting. By combining a general-purpose backbone with a scalable contextual pattern bank, STBP efficiently mitigates catastrophic forgetting while capturing dynamic spatio-temporal correlations. It adapts to evolving urban data without retraining from scratch, making it suitable for real-time applications.

Validated on multiple datasets, STBP demonstrates strong continual learning capabilities. Nevertheless, STBP currently supports continual learning in a single-task setting. In the future, we plan to extend its application to cross-domain continual spatio-temporal forecasting, which will be a crucial step towards developing a foundational spatio-temporal model.

## Acknowledgments

This work was partly supported by the National Key Research and Development Program of China under Grant 2022YFB4501704, the National Natural Science Foundation of China under Grant 72342026, and Fundamental Research Funds for the Central Universities under Grant 2024-6-ZD-02.

## References

Lei Bai, Lina Yao, Can Li, Xianzhi Wang, and Can Wang. Adaptive graph convolutional recurrent network for traffic forecasting. volume 33, pp. 17804–17815, 2020.

Lucas Caccia, Eugene Belilovsky, Massimo Caccia, and Joelle Pineau. Online learned continual compression with adaptive quantization modules. In *International conference on machine learning*, pp. 1240–1250. PMLR, 2020.

Chao Chen, Karl Petty, Alexander Skabardonis, Pravin Varaiya, and Zhanfeng Jia. Freeway performance measurement system: mining loop detector data. *Transportation research record*, 1748(1):
96–102, 2001.

Wei Chen and Yuxuan Liang. Expand and compress: Exploring tuning principles for continual spatio-temporal graph forecasting. In The Thirteenth International Conference on Learning Representations, 2025.

Xu Chen, Junshan Wang, and Kunqing Xie. Trafficstream: A streaming traffic flow forecasting framework based on graph neural networks and continual learning. In Zhi-Hua Zhou (ed.), Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21, pp. 3620–3626, 8 2021. doi: 10.24963/ijcai.2021/498. URL https://doi.org/10.24963/ ijcai.2021/498.

Zheng Dong, Renhe Jiang, Haotian Gao, Hangchen Liu, Jinliang Deng, Qingsong Wen, and Xuan Song. Heterogeneity-informed meta-parameter learning for spatiotemporal time series forecasting. In *Proceedings of the 30th ACM SIGKDD conference on knowledge discovery and data mining*, pp. 631–641, 2024.

Yuchen Fang, Yanjun Qin, Haiyong Luo, Fang Zhao, and Kai Zheng. Stwave+: A multi-scale efficient spectral graph attention network with long-term trends for disentangled traffic flow forecasting. IEEE Transactions on Knowledge and Data Engineering, 2023.

Yuchen Fang, Hao Miao, Yuxuan Liang, Liwei Deng, Yue Cui, Ximu Zeng, Yuyang Xia, Yan Zhao, Torben Bach Pedersen, Christian S. Jensen, Xiaofang Zhou, and Kai Zheng. Unraveling Spatio-Temporal Foundation Models Via the Pipeline Lens: A Comprehensive Review . *IEEE*
Transactions on Knowledge & Data Engineering, (01):1–24, January 2026. ISSN 1558-2191.

doi: 10.1109/TKDE.2026.3651536. URL https://doi.ieeecomputersociety.org/ 10.1109/TKDE.2026.3651536.

Haotian Gao, Renhe Jiang, Zheng Dong, Jinliang Deng, Yuxin Ma, and Xuan Song. Spatialtemporal-decoupled masked pre-training for spatiotemporal forecasting. In Kate Larson (ed.), Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, IJCAI-24, pp. 3998–4006. International Joint Conferences on Artificial Intelligence Organization, 8 2024. doi:
10.24963/ijcai.2024/442. URL https://doi.org/10.24963/ijcai.2024/442. Main Track.

Danlei Hu, Lu Chen, Hanxi Fang, Ziquan Fang, Tianyi Li, and Yunjun Gao. Spatio-temporal trajectory similarity measures: A comprehensive survey and quantitative study. IEEE Transactions on Knowledge and Data Engineering, 36(5):2191–2212, 2023.

Jiahao Ji, Wentao Zhang, Jingyuan Wang, and Chao Huang. Seeing the unseen: Learning basis confounder representations for robust traffic prediction. In *Proceedings of the 31st ACM SIGKDD*
Conference on Knowledge Discovery and Data Mining V. 1, pp. 577–588, 2025.

Jiawei Jiang, Chengkai Han, Wayne Xin Zhao, and Jingyuan Wang. Pdformer: Propagation delayaware dynamic long-range transformer for traffic flow prediction. In Proceedings of the AAAI Conference on Artificial Intelligence, 2023a.

Renhe Jiang, Zhaonan Wang, Jiawei Yong, Puneet Jeph, Quanjun Chen, Yasumasa Kobayashi, Xuan Song, Shintaro Fukushima, and Toyotaro Suzumura. Spatio-temporal meta-graph learning for traffic forecasting. In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 37, pp. 8078–8086, 2023b.

Ming Jin, Huan Yee Koh, Qingsong Wen, Daniele Zambon, Cesare Alippi, Geoffrey I Webb, Irwin King, and Shirui Pan. A survey on graph neural networks for time series: Forecasting, classification, imputation, and anomaly detection. *IEEE Transactions on Pattern Analysis and* Machine Intelligence, 2024.

Angelos Katharopoulos, Apoorv Vyas, Nikolaos Pappas, and Franc¸ois Fleuret. Transformers are rnns:
Fast autoregressive transformers with linear attention. In International Conference on Machine Learning (ICML), pp. 5156–5165. PMLR, 2020.

Weiyang Kong, Ziyu Guo, and Yubao Liu. Spatio-temporal pivotal graph neural networks for traffic flow forecasting. *Proceedings of the AAAI Conference on Artificial Intelligence*, 38(8):8627–8635, Mar. 2024. doi: 10.1609/aaai.v38i8.28707. URL https://ojs.aaai.org/index.php/ AAAI/article/view/28707.

Rahul Kumar, Manish Bhanu, Joao Mendes-Moreira, and Joydeep Chandra. Spatio-temporal predic- ˜
tive modeling techniques for different domains: a survey. *ACM Computing Surveys*, 57(2):1–42, 2024.

Sanghyun Lee and Chanyoung Park. Continual traffic forecasting via mixture of experts. arXiv preprint arXiv:2406.03140, 2024.

Fuxian Li, Jie Feng, Huan Yan, Guangyin Jin, Fan Yang, Funing Sun, Depeng Jin, and Yong Li.

Dynamic graph convolutional recurrent network for traffic prediction: Benchmark and solution.

ACM Transactions on Knowledge Discovery from Data, 17(1):1–21, 2023.

Yaguang Li, Rose Yu, Cyrus Shahabi, and Yan Liu. Diffusion convolutional recurrent neural network:
Data-driven traffic forecasting. In *International Conference on Learning Representations (ICLR)*,
2018.

Aoyu Liu and Yaying Zhang. An efficient spatial-temporal transformer with temporal aggregation and spatial memory for traffic forecasting. *Expert Systems with Applications*, 250:123884, 2024a.

Aoyu Liu and Yaying Zhang. Spatial–temporal dynamic graph convolutional network with interactive learning for traffic forecasting. *IEEE Transactions on Intelligent Transportation Systems*, 2024b.

Aoyu Liu and Yaying Zhang. Crossst: An efficient pre-training framework for cross-district pattern generalization in urban spatio-temporal forecasting. In 41th IEEE International Conference on Data Engineering, 2025.

Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, and Rui Zhao. Spatialtemporal large language model for traffic prediction. In *25th IEEE International Conference on* Mobile Data Management (MDM), pp. 31–40, 2024a.

Chenxi Liu, Kethmi Hirushini Hettige, Qianxiong Xu, Cheng Long, Shili Xiang, Gao Cong, Ziyue Li, and Rui Zhao. ST-LLM+: Graph enhanced spatio-temporal large language models for traffic prediction. *IEEE Transactions on Knowledge and Data Engineering*, 37(8):4846–4859, 2025a.

Chenxi Liu, Qianxiong Xu, Hao Miao, Sun Yang, Lingzheng Zhang, Cheng Long, Ziyue Li, and Rui Zhao. TimeCMA: Towards llm-empowered multivariate time series forecasting via cross-modality alignment. In *AAAI*, volume 39, pp. 18780–18788, 2025b.

Hangchen Liu, Zheng Dong, Renhe Jiang, Jiewen Deng, Q Chen, and X Song. Staeformer: Spatiotemporal adaptive embedding makes vanilla transformers sota for traffic forecasting. In *Proceedings* of the 32nd ACM International Conference on Information and Knowledge Management (CIKM), pp. 21–25, 2023a.

Xu Liu, Yutong Xia, Yuxuan Liang, Junfeng Hu, Yiwei Wang, Lei Bai, Chao Huang, Zhenguang Liu, Bryan Hooi, and Roger Zimmermann. Largest: A benchmark dataset for large-scale traffic forecasting. In *Advances in Neural Information Processing Systems*, 2023b.