# XTransfer: Modality-Agnostic Few-Shot Model Transfer for Human Sensing at the Edge

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2, 4

## Abstract
Deep learning for human sensing on edge systems presents significant potential for smart applications. However, its training and development are hindered by the limited availability of sensor data and resource constraints of edge systems. While transferring pre-trained models to different sensing applications is promising, existing methods often require extensive sensor data and computational resources, resulting in high costs and poor adaptability in practice. In this paper, we propose XTransfer, a first-of-its-kind method enabling modality-agnostic, few-shot model transfer with resource-efficient design. XTransfer flexibly uses single or multiple pre-trained models and transfers knowledge across different modalities by (i) model repairing that safely mitigates modality shift by adapting pre-trained layers with only few sensor data, and (ii) layer recombining that efficiently searches and recombines layers of interest from source models in a layer-wise manner to create compact models. We benchmark various baselines across diverse human sensing datasets spanning different modalities. Comprehensive results demonstrate that XTransfer achieves state-of-the-art performance while significantly reducing the costs of sensor data collection, model training, and edge deployment.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces a framework for modality-agnostic, few-shot model transfer tailored for human sensing on edge devices. The core contributions are a splice repair removal pipeline that mitigates modality shift by aligning latent feature distributions of target sensor data with pre-trained source models using an anchor-based loss in a reduced PCA space, and a layer wise search mechanism that efficiently searches and recombines useful layers from single or multiple source models to construct a compact, high-performance target model.

### Strengths
Tackles a practical problem at the intersection of few-shot learning, cross-modal transfer, and edge AI. The proposed method's ambition is to leverage readily available pre-trained models from vastly different modalities (e.g., image, text) for specialized sensing tasks. The quality of experimental evaluation is good.

### Weaknesses
The method's reliance on mean magnitude of channels and its s-score as the primary metric for guiding layer repairing and selection feels under-justified. While it  is presented as a lightweight metric, its suitability for capturing feature discriminability across drastically different modalities (e.g., vision to IMU) is not intuitively clear, and a more thorough justification or comparison against other feature distribution metrics (e.g., MMD) would strengthen this core design choice. 

The proposed approach is also quite complex, involving multiple components, stages, and hyperparameters (e.g., PCA dimensionality, search parameters), which could pose challenges for reproducibility and practical implementation. 

Finally, LWS could face scalability issues as the number and depth of source models increase and the robustness of the proposed $rate^{est}$ model for the pre-search check is not fully explored, especially in highly dissimilar transfer settings.

### Questions
The "Model Repairing" component centers on aligning feature spaces to minimize MMC shift. Could you elaborate on the intuition for using a channel-magnitude metric like MMC for cross-modality transfer, where feature representations are fundamentally different? Have you experimented with alternative feature alignment techniques, such as adversarial alignment or MMD, and how do they compare?

The LWS recombines layers sequentially from source models. How does this mechanism handle non-sequential architectural elements, such as the skip connections in ResNet architectures? Are these connections discarded, or does your method have a way to preserve or reconstruct them in the final compact model?

The pre-search check's efficiency depends on an exponential growth model for the repair rate. How was this specific model form chosen, and how robust is the search process if the actual repair rate for a given source-target pair deviates a lot from assumed exponential trend?

Table 4 shows that in the challenging 3-shot setting, XTransfer's accuracy is slightly below the oracle baseline on some tasks. Does this point to a fundamental limit on the minimum data required for stable alignment, and could this gap be closed by integrating few-shot data augmentation techniques into the SRR pipeline?

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
This paper introduces a modality-agnostic few-shot transfer learning framework tailored for resource-constrained human-sensing applications on edge devices.
The method leverages pre-trained models as sources and combines layer repairing to mitigate modality shift with layer-wise recombination to select only beneficial layers, thereby producing compact, efficient models.
The authors evaluated the proposed method across several sensing datasets and showed state-of-the-art accuracy while reducing sensor data requirements, training cost, and deployment resource overhead.

### Strengths
- The paper identifies a timely and well-motivated challenge in human-sensing systems and tackles few-shot cross-modality transfer on resource-constrained edge devices.
- The proposed XTransfer framework integrates a structured SRR pipeline for modality repair with a principled layer-wise recombination strategy. The design addresses both representation alignment and parameter efficiency, demonstrating a thoughtful mechanism for reusing heterogeneous pre-trained models.
- The study evaluates the approach across multiple sensing modalities, diverse benchmarks, and real edge-device settings. Results consistently show improvements in accuracy-resource trade-offs and training efficiency, providing convincing evidence of the method's scalability and practicality for deployment.

### Weaknesses
- The paper would benefit from a discussion of failure cases, sensitivity to noisy or highly heterogeneous sensor data, and robustness under severe domain shifts
- The SRR and layer-wise search procedures introduce methodological complexity, and the paper does not fully quantify the tuning burden, search overhead under diverse hardware constraints, or potential stability issues when scaling to larger sets of heterogeneous source models.
- The evaluation focuses primarily on cross-modality human-sensing tasks, and comparison to broader transfer paradigms (e.g., recent foundation-model or prompt-based adaptation techniques) is limited.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces XTransfer, a cross-modal adaptation framework bridging pretrained models and sensing applications through layer manipulations. The authors propose two components, a Spliece-Repair-Removal pipeline that adapts pretrained layers to new sensor modalities using limited sensor data, and a Layer Wise Search that recombines effective layers for a compact, efficient model. The paper conducts thorough evaluations on 8 datasets and shows improved performance and resource efficiency under limited-sample scenarios. The appendix also contains ablation studies to understand the significance of the components proposed.

### Strengths
1. The attempt to reuse pre-trained models from heterogeneous modalities such as images and text to accelerate sensor-domain adaptation is novel. 
2. Extending few-shot learning to a modality-agnostic context is novel and has great applicability.
3. Evaluation across multiple modalities and domains is comprehensive.
4. Overall the motivation is strong and the proposed method outperforms the baselines

### Weaknesses
- The paper needs significant work on the presentation for better clarity. Some examples below:
    - The term channels used during the removal and repair stages was not clarified and could be misleading, given its context in signal processing.
    - Preliminary motivation is unclear. Figure 3 shows relationships among MMC, accuracy, and other metrics, but fails to specify the details like the models and domains. Moreover, sensing as a modality remains underspecified. Baselines plotted in Figure 3 are never introduced until later sections.
    - The methods proposed (SRR and LWS) modules are described very densely with poor structures with little intuition or top-down explanation. Figures are overcrowded and fail to clearly depict information flow across stages and are very far away from where it is referenced.
- The authors should also compare self-supervised methods. Current work seems to be evaluating on supervised pretrained source models. However, self-supervised models already show great generalizability and cross-domain transfer capabilities. This could improve the impact of the work.
- The authors reported the training-time statistics but do not discuss the convergence rate, especially given the inclusion of a generator-based repair module. The paper does not compare convergence speed with standard SSL or linear-probe methods, which makes it uncertain whether the proposed system actually converges faster or simply trains less data per step.
- The LWS module is described as a search process for selecting effective layers over NAS, but it lacks a comparison against any established search or pruning methods. So it is hard to determine the significance of prior search works.
- The newest baselines, SemiCMT, seem to be a self-supervised cross-modal alignment framework that would require paired data. It is confusing how SemiCMT was trained given there is no cross-modal pairs between source and target domains. It is unclear how the baselines are trained for fair comparison
- It is unclear on the exact number of samples used for each source dataset, reporting only the number of classes and input shapes. Since source data scale strongly affects transfer quality, it is unclear on the cross-modality transfer performance, since image source datasets usually have a larger scale and are likely to have higher transfer performance compared to other modality source datasets. So it i s unclear on the validity of the conclusion in 6.2 Impact of different sources.
- Most of the baselines are relatively old (19 - 22), the most recent baselines are SemiCMT which was designed for cross-modal alignment that requires multimodal pair and GPT2 which is a generative model not suitable for the downstream classification.

### Questions
Please see the weakness for most of the concerns. Some questions for authors to discuss are:
- Differences between area A and area B trends (where MMC correlates differently with accuracy) are not explained. Can authors provide more clarification on this?
- Since SemiCMT requires multimodal pairing, how is it adapted to the unpaired cross-domain case where source and target modalities differ completely?
- What are the scales of the dataset in terms of number of samples?
- Is there any established search or pruning algorithms (e.g., NAS, lottery-ticket, or L2-pruning) used for comparison?
- Most of the time the target domain might have more than just 10 samples per class, what happens when there are more target domain samples, would XTransfer still have the competing performance?
- Can authors elaborate more on comparison against SSL finetune with additional input head and downstream head for cross-modal and cross-domain adaptation?

### Soundness
2

### Presentation
1

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes XTransfer, addressesing the data scarcity and resource constraints of human sensing on edge devices by enabling modality-agnostic, few-shot model transfer. It repurposes pre-trained models for diverse sensing modalities using very few labeled sensor samples. Its core pipeline includes two key components: (1) Model Repairing via a Splice-Repair-Removal (SRR) pipeline—aligning latent feature distributions across modalities; (2) Layer Recombining via Layer-Wise Search (LWS) control—selecting and recombining only useful repaired layers to build compact models. 
Experiments on 8 source datasets (image/text/audio/sensing) and 7 target datasets show XTransfer outperforms SOTA baselines

### Strengths
- Novel Modality-Agnostic Paradigm: Unlike prior transfer methods (limited to same-modality or paired cross-modal data), XTransfer achieves transferring knowledge from image/text pre-trained models to sensing modalities with few labeled data. This setting resolves the high cost of sensing data collection and leverages public pre-trained models as "free" knowledge sources.

- Theoretically Grounded and Empirically Valid Method Mechanism: The SRR pipeline’s design (PCA orthogonal space, anchor-based loss, class pairing) is justified by Transformer layer dynamics. It effectively mitigates modality shift as evidenced by experiments.

- Resource-Efficient Design for Edge Deployment: LWS control’s layer selection and pre-search check reduce model size by 2.4–16.5× in FLOPs vs. source backbones, while maintaining SOTA accuracy. On edge devices, latency is cut by 1.4–21×, making it practical for resource-constrained human sensing.

### Weaknesses
- Dependence on PCA for Feature Alignment: XTransfer relies on linear PCA to reduce dimensionality and align features. However, the concern is that PCA fails to capture non-linear relationships between source and target modalities (e.g., text embeddings vs. Doppler radar signals), which may limit performance in highly dissimilar cross-modality scenarios (e.g., text → ECG). 

- Brittleness in Extremely Low-Shot Settings: While XTransfer performs well in 5–10-shot scenarios, it struggles with 3-shot settings—e.g., accuracy lags the oracle baseline on HHAR/Gesture datasets. This raises concerns for ultra-scarcity sensing tasks (e.g., rare medical conditions).

- Homogeneous Source Model Assumption: The framework assumes pre-trained source models have homogeneous architectures (e.g., all ResNet variants). Extending to heterogeneous backbones is not fully validated—layer recombination across structurally diverse models (e.g., CNN vs. Transformer) may break MMC shift estimation and layer-wise dependence, limiting scalability to multi-modal source pools.

- the writing can be further improved for clarity. Too many abbreviated terms,such as MMC, may weakean readability.

### Questions
- How would XTransfer perform with non-linear feature alignment methods? 

- Can XTransfer be extended to ultra-low-shot (1–2-shot) scenarios?

- How does XTransfer handle heterogeneous source models?

### Soundness
2

### Presentation
2

### Contribution
2
