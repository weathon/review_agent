# Skillful Kilometer-Scale Regional Weather Forecasting via Global and Regional Coupling

- Decision: Reject
- Scores: 6, 4, 4

## Abstract
Data-driven weather models have advanced global medium-range forecasting, yet high-resolution regional prediction remains challenging due to unresolved multiscale interactions between large-scale dynamics and small-scale processes such as terrain-induced circulations and coastal effects. This paper presents a **global-regional coupling framework** for kilometer-scale regional weather forecasting that synergistically couples a pretrained Transformer-based global model with a high-resolution regional network via a novel bidirectional coupling module, **ScaleMixer**. ScaleMixer dynamically identifies meteorologically critical regions through adaptive key-position sampling and enables cross-scale feature interaction through dedicated attention mechanisms. The framework produces forecasts at  $0.05^\circ$ ($\sim 5 \mathrm{km}$ ) and 1-hour resolution over China, significantly outperforming operational NWP and AI baselines. It exhibits exceptional skill in capturing fine-grained phenomena such as orographic wind patterns, Foehn warming, and coastal transitions during typhoon events, demonstrating effective global-scale coherence with high-resolution fidelity. The code is available at https://anonymous.4open.science/r/ScaleMixer-6B66.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the challenge of high-resolution regional weather forecasting, where global forecasts are typically used as inputs in a decoupled manner that overlooks cross-scale interactions, leading to less accurate predictions. To overcome this, the authors propose a global–regional coupling framework specifically designed for high-resolution regional forecasting. This framework includes a module called ScaleMixer, which identifies spatial regions with strong multiscale interactions and learns bidirectional feature encoding between global and regional tokens in these areas. The experimental analysis includes two complementary settings, hindcast and operational forecasting, both demonstrating the effectiveness of the proposed architecture.

Specifically, the authors formulate the forecasting problem as developing a hybrid global–regional weather forecasting framework that extends the capabilities of a pretrained global network by leveraging interactions between large-scale (global) atmospheric dynamics and small-scale (regional) weather effects. The pretrained global model, M_global, is a ViT-based architecture trained on the ERA5 reanalysis dataset. The regional model,  M_regional, uses the same architecture with certain modifications. The ScaleMixer module then models the interaction between these two networks by learning their relationships and performing token-level encoding based on them, effectively prioritizing regions with strong cross-scale dependencies.

### Strengths
- The authors clearly identify the gap in the current literature on regional weather forecasting. The related work section provides an up-to-date review of both global and regional forecasting studies. The motivation behind the proposed architecture is clearly presented as a mechanism to dynamically identify cross-scale interaction regions and enable bidirectional feature encoding between global and regional tokens through a coupling framework. The overall structure of the paper is clear, well-organized, and easy to follow.
- The experimental analysis (Table 1 and Figure 2) clearly demonstrates the superiority of the proposed model over other methods by a significant margin. Including the results of M_global and M_regional further highlights the need for a coupled architecture that leverages both global and regional information, hence the need for ScaleMixer. Case studies on orographic-induced wind and temperature forecasting, and extreme event prediction further support the effectiveness of the proposed approach. Visualizations of forecasts for various metrics, such as temperature and surface pressure, also show that ScaleMixer adapts well to different scenarios rather than performing well in only a single setting.

### Weaknesses
- For an experimental work like this, the ablation study is a crucial component, as it should strongly justify the need for each distinct part of the architecture. While the paper provides clear motivation for each component in earlier sections, for instance, the need for a ScaleMixer like structure, the choice of its exact architecture would be much more convincing if supported by a dedicated ablation study.
- Similarly, the proposed architecture includes multiple components that depend on various hyperparameters, such as patch size and the number of encoder layers. The authors neither specify what kind of validation—if any—was used to select these hyperparameters nor present results using alternative settings. Including these details would experimentally strengthen the evidence for the model’s efficiency and its superiority over other baseline methods.
- The authors state that the code for the proposed architecture and experiments is available at the provided URL for reproducibility. However, the URL does not appear to contain any implementation.

### Questions
- See the weakness section for my detailed comments.

### Soundness
2

### Presentation
3

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
This paper presents a novel deep learning framework for kilometer-scale regional weather forecasting, a task that remains a significant challenge due to complex multiscale interactions. The proposed model synergistically couples a pre-trained, low-resolution global Transformer model with a high-resolution regional network. The core innovation is the ScaleMixer module, a bidirectional coupling mechanism that dynamically identifies meteorologically critical regions and facilitates efficient information exchange between the global and regional scales using a two-stage attention process. The framework achieves state-of-the-art performance on 0.05° (~5km), 1-hour forecasts over a challenging region in China, significantly outperforming both operational numerical weather prediction (NWP) systems and existing AI baselines.

### Strengths
- Novel and Effective Architecture: The ScaleMixer module is a clever and well-designed innovation. Its use of adaptive key-position sampling makes the cross-scale attention computationally feasible while focusing on meteorologically important areas. The bidirectional information flow is a crucial element that distinguishes it from simpler downscaling or one-way coupling approaches.
- State-of-the-Art Results and Rigorous Evaluation: The model demonstrates clear superiority over very strong baselines, including one of the world's best operational NWP systems. The dual hindcast and operational evaluation setup provides a high degree of confidence in the results.
- Strong Qualitative Evidence: The case studies on orographic wind, the Foehn effect, and a typhoon event are compelling. They visually demonstrate that the model is not just improving on aggregate metrics but is genuinely capturing complex, fine-grained physical phenomena that coarser models miss.

### Weaknesses
- Architectural Complexity: The overall system is highly complex, involving a large pre-trained global model, a regional model, and the intricate ScaleMixer module, with a multi-stage training process. This complexity may pose a barrier to reproduction and further analysis by other researchers.
- Limited Ablation Studies: While the paper compares against standalone global and regional models (which serves as a high-level ablation), it would benefit from more fine-grained ablation studies on the ScaleMixer itself. For instance, quantifying the impact of the adaptive key position selection (vs. a fixed grid) or the bidirectional feedback (vs. a unidirectional flow) would further solidify the importance of these specific design choices.
- Geographical Generalizability: The experiments are concentrated on a single, albeit challenging, geographical region (China). While this is a strong proof of concept, a discussion on the potential challenges or necessary adaptations for applying the framework to other regions with different dominant weather patterns (e.g., the tropics, polar regions) would be beneficial.

### Questions
- Could you provide ablations on the key components of ScaleMixer, such as the adaptive key position selection (vs. fixed or random) and the bidirectional feedback mechanism (vs. one-way)? This would help isolate the performance gains attributable to each architectural innovation.
- The model is very large (1.07B parameters). Could you provide a comparison of the computational cost (e.g., FLOPs, wall-clock inference time) against the NWP baseline (IFS-HRES) for generating a single forecast? This is critical for understanding its potential for operational deployment.
- Have you considered how the framework might perform in other geographical regions where the dominant physical processes might differ (e.g., large-scale, flat convection in the US Great Plains vs. the orographically-driven dynamics shown here)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper presents a global-regional coupled model at the kilometer scale. Establishing the link between global and regional weather forecasting represents a critical challenge, and the stated motivation is noteworthy. The experiments provide some evidence of the advantages of the proposed ScaleMixer approach. However, the innovation appears limited. The primary concept parallels existing research, and prior contributions seem insufficiently acknowledged, with the work potentially presented as novel.

### Strengths
1. The research direction is promising.
2. The writing quality is good.

### Weaknesses
1. The author makes an inaccurate explanation for existing regional forecasting models, which despicts in Line 97-101 (Limited area modeling methods (Nipen et al., 2024; Gao et al., 2025) employ GNN architectures with stretched-grid to make global weather forecasts, imposing denser grids and higher weights over specific areas to achieve local predictions with high-spatial resolution. However, these models use global forecasts and their shallow features as an initial context for regional forecasting, but ignore their dynamic interactions.). As far as I know, for (Nipen et al., 2024), it makes a specific design for the interection of global and regional forecasts. And for (Gao et al., 2025), it proposes a neural nested grid method to achieve the dynamic interactions. 
2. Why didn’t you make the comparison with [1]? All of them are targeting kilometer-scale weather forecasting.
3.  As for the innovation, I can not find  obvious innovation compared with OneForecast[2], all of them first pretrain a global model, and in the next stage, the global model is frozen, and a regional model is trained.
4. Missing comparison with regional forecasting models, such as DDM [3], Graph-EFM[4],  OneForecast[2], and [1].
5. For global forecasting, it lacks the comparison with WeatherBench2’s baselines, such as Pangu, Fuxi, Graphcast, and NeuralGCM, etc.



[1] Building Machine Learning Limited Area Models: Kilometer-Scale Weather Forecasting in Realistic Settings

[2] OneForecast: a universal framework for global and regional weather forecasting

[3] Regional data-driven weather modeling with a global stretched-grid

[4] Probabilistic Weather Forecasting with Hierarchical Graph Neural Networks

### Questions
See Weaknesses

### Soundness
2

### Presentation
2

### Contribution
2
