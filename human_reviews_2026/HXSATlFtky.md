# The Other Side of the Coin: Unveiling the Downsides of Model Aggregation in Federated Learning from a Layer-peeled Perspective

- Avg Score: 4.50
- Decision: Reject
- Scores: 6, 4, 2, 6

## Abstract
In federated learning (FL), model aggregation plays a central role in enabling decentralized knowledge sharing.
However, it is often observed that the aggregated model underperforms on local data until after several rounds of local training.
This temporary performance drop can potentially slow down the convergence of the FL model. Prior work regards this performance drop as an inherent cost of knowledge sharing among clients and does not give it special attention. While some studies directly focus on designing techniques to alleviate the issue, its root causes remain poorly understood. To bridge this gap, we construct a framework that enables layer-peeled analysis of how feature representations evolve during model aggregation in FL.  It focuses on two key aspects: (1) the intrinsic quality of extracted features, and (2) the alignment between features and their subsequent parameters---both of which are critical to downstream performance. Using this framework, we first investigate what model aggregation does to the internal feature extraction process. Our analysis reveals that aggregation degrades feature quality and weakens the coupling between intermediate features and subsequent layers, both of which are well shaped during local training. More importantly, this degradation is not confined to specific layers but progressively accumulates with network depth---a phenomenon we term Cumulative Feature Degradation (CFD).
CFD severely impairs the quality of penultimate-layer features, ultimately compromising the model's decision-making capacity.
Next, we examine how key FL settings---such as aggregation frequency---can exacerbate or alleviate the negative effects of model aggregation. Finally, we revisit several commonly used strategies, such as initialization from pretrained models, and explain \textbf{why} they are effective through layer-peeled analysis. To the best of our knowledge, this is the first systematic study of model aggregation in FL from a layer-peeled feature extraction perspective, potentially paving the way for the development of more effective FL algorithms.
The code is available at:https://anonymous.4open.science/r/ICLR_14921_Code-3565.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates model aggregation in FL from a layer-wise feature perspective by proposing a layer-peeled analysis framework for a more interpretable lens to understand the internal dynamics of FL. The analysis reveals a key phenomenon termed Cumulative Feature Degradation (CFD), and the study further examines how different FL settings influence this degradation during model aggregation.

### Strengths
- The topic is relevant and important. The construction of a layer-peeled feature analysis framework is helpful.
- The findings of CFD help explain why the performance drop is so pronounced and why it is a fundamental challenge in aggregating deep models.
- The analysis covers multiple datasets and model architectures to support the findings.

### Weaknesses
- The current analysis is primarily empirical, relying on experimental metrics. The paper would be strengthened by incorporating theoretical analysis to support or generalize the empirical findings.
- While the abstract and introduction highlight aggregation frequency as a key factor, the corresponding analysis is put into appendix. Including it in the main text along with a more detailed discussion would be better.
- Although the paper successfully diagnoses a key issue (CFD), it does not propose concrete solutions or algorithmic adjustments inspired by the insights. The claim in the abstract that the work “potentially paves the way” for better FL algorithms would be more convincing if accompanied by specific, testable hypotheses or design principles.
- The figures and captions in the supplementary section can be further elaborated.

### Questions
Please see the weakness part.

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
This paper presents a layer-peeled analysis framework to investigate performance degradation after model aggregation in federated learning (FL). The study shows that feature variance and feature-parameter alignment deteriorate as network depth increases, a phenomenon the paper refers to as cumulative feature degradation (CFD). The paper further demonstrates that model aggregation can improve feature generalization across clients. Finally, it analyzes how existing FL strategies mitigate the effects of CFD to achieve improved performance.

### Strengths
- Overall, the paper is well-written, and the figures and explanations are clear and easy to follow. 

- The paper provides a systematic set of metrics for analyzing the dynamics of features and model parameters in FL settings.

### Weaknesses
- The analysis mainly focuses on the proposed analytical metrics without presenting accompanying accuracy trends to support the findings. While the paper suggests that model aggregation may degrade performance, it does not clearly demonstrate how the performance drops would correlate with the reported feature and parameter metrics and their dynamics.

- At Lines 273-279, the paper briefly introduces and defines CFD as the larger relative changes in the metrics as network depths increase. However, this definition is somewhat vague and lacks direct evidence that the degradation is indeed cumulative, since the observed relative changes may be influenced by various factors. For example, if performance is already low at earlier layers, the relative change may appear smaller simply because there is limited room for further degradation.

- The paper introduces CFD and uses it to analyze feature and parameter dynamics in FL, as well as to interpret the behavior of existing FL approaches. However, it is not entirely clear what concrete insights CFD offers that would meaningfully guide the design of future FL methods or lead to further advances in the field.

### Questions
Besides the weakness shown in the above section, please also see the following questions: 

Q1: In Figures 2 and 3, the relative changes in feature variance increase with network depth. However, the feature variances in the shallow layers are already not performing well (e.g., large within-class variance and small between-class variance for “L1” in Figures 2(a) and 2(c)). In this case, can we still conclude that feature degradation becomes more severe in deeper layers? The smaller relative change observed in shallow layers may simply be due to their initially poor performance, rather than indicating less degradation. 

Q2: In Figure 8, it seems that the personalization method FedBN still exhibits larger relative changes in the deeper layers than FedAvg. Does this imply that FedBN is less effective in mitigating CFD? Additionally, what is the accuracy comparison between FedAvg and FedBN?

Q3: At Lines 14-16 in the abstract, the paper states that performance drops after aggregation can potentially slow down the convergence of FL. However, in Section 4.4, the results indicate that model aggregation improves generalization. Why would improved model generalization hinder convergence? Does this imply that without aggregation, the model would converge more quickly but to an overfitted local minimum? If so, would slower convergence in this case actually be preferable?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates why model averaging in federated learning (FL) often causes a temporary drop in clients’ local performance after aggregation. The authors identify that existing work treats this post-aggregation degradation as an inherent cost without explaining its internal mechanisms. To address this, they propose a layer-peeled analysis framework that examines how aggregation alters feature representations and their alignment with subsequent layers, introducing the concept of Cumulative Feature Degradation (CFD), a depth-accumulating degradation of feature quality and feature-parameter alignment. Through empirical analysis, they show that aggregation increases within-class variance, decreases between-class variance, and disrupts alignment between penultimate features and classifiers, while also improving out-of-distribution generalization.

### Strengths
1. The figure illustrating the layer-wise performance trend is well-presented and effectively supports the analysis.

2. The experimental setup is described with sufficient clarity and detail to ensure reproducibility.

### Weaknesses
1. Limited novelty compared to prior layer-wise/feature-alignment analyses. Prior work already diagnoses aggregation-induced feature/layer misalignment and layer-dependent behavior, and studies when layer-wise averaging or alignment helps (e.g., Fed2 [1] aligns features across clients; pFedLA [2] learns layer-wise aggregation analysis in personalized FL setting; FedFA provides detailed analysis of latent feature statistics and provide a feature alignment method; Layer-wise Linear Mode Connectivity [4] shows layers often admit linear connectivity with thorough analysis of the layer-wise parameter dynamics in model aggregation). I don’t clearly see what new insight this paper adds beyond those feature alignment analyses. 

2. CFD seems like a correlate, not a fundamental driver. The experiments mainly establish correlations between CFD metrics and accuracy without causal interventions; the “cumulative” phrasing is also puzzling because aggregation is a weight-averaging step (no depth-wise propagation), and the depth trend likely reflects local training signals rather than averaging per se.

3. Explanation is generic and widely known. The argument reduces to “within-class variance rises and between-class separation falls after averaging,” which mirrors established neural-collapse/feature-separation results [5, 6] in standard deep nets; please clarify what is federation-specific beyond these generic patterns or provide theory linking heterogeneity of federated learning setting.

[1] Yu, Fuxun, et al. "Fed2: Feature-aligned federated learning." Proceedings of the 27th ACM SIGKDD conference on knowledge discovery & data mining. 2021.  
[2] Ma, Xiaosong, et al. "Layer-wised model aggregation for personalized federated learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022. 
[3] Zhou, Tianfei, and Ender Konukoglu. "FedFA: Federated Feature Augmentation." The Eleventh International Conference on Learning Representations.  
[4] Adilova, Linara, et al. "Layer-wise linear mode connectivity." The Twelfth International Conference on Learning Representations.  
[5] Papyan, Vardan, X. Y. Han, and David L. Donoho. "Prevalence of neural collapse during the terminal phase of deep learning training." Proceedings of the National Academy of Sciences 117.40 (2020): 24652-24663.  
[6] Parker, Liam, et al. "Neural collapse in the intermediate hidden layers of classification neural networks." arXiv preprint arXiv:2308.02760 (2023).

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the temporary performance drop seen in Federated Learning (FL) after models are aggregated, using a novel "layer-peeled" analysis framework to understand the root causes. The authors identify a phenomenon called Cumulative Feature Degradation (CFD), where aggregation progressively degrades feature quality and disrupts the alignment between features and parameters as network depth increases. This degradation, especially the mismatch between the final features and the classifier, is pinpointed as the main cause of the performance drop. Despite this downside, the study confirms that aggregation is vital for improving model generalization and preventing overfitting to local client data. The paper also uses this framework to explain why common FL solutions work, showing that methods like parameter personalization, pre-trained initialization, and classifier fine-tuning are effective because they successfully mitigate the CFD effect

### Strengths
1. The authors rigorously demonstrate that the negative impact of aggregation is not a uniform hit but a compounding problem that progressively accumulates with network depth.

2. The study offers a balanced perspective. It not only identifies the downsides of aggregation (CFD) but also validates its crucial upside, showing that aggregation is what enables the model to create more generalizable features and mitigate local overfitting.

3. The paper introduces a "layer-peeled" analysis framework  that moves beyond standard accuracy or loss metrics.

### Weaknesses
1. The experimental setup involves a very small number of clients (e.g., 4 clients for PACS, 6 for DomainNet). This is not representative of typical cross-device FL scenarios, which can involve hundreds, thousands, or even millions of clients. The dynamics of averaging four or six models may be very different from averaging thousands, and it remains an open question whether the severity and behavior of CFD would scale, diminish, or change entirely in a massively federated setting. 

2. The paper's conclusions about "model aggregation" are almost exclusively based on analyzing the FedAvg algorithm, which uses simple parameter-wise averaging. While FedAvg is a foundational baseline, the paper does not investigate whether the Cumulative Feature Degradation (CFD) phenomenon persists in more advanced FL algorithms designed specifically to combat aggregation problems (like FedProx, SCAFFOLD, or FedDyn). It's possible that CFD is a specific artifact of the naive FedAvg approach rather than an unavoidable downside of all model aggregation in FL

3. All experiments are conducted on image classification datasets (Digit-Five, PACS, and DomainNet) using standard vision architectures (CNNs and ViT) . The findings, while significant for computer vision, cannot be assumed to generalize to other major applications of FL. It is unknown if CFD manifests similarly in fundamentally different tasks, such as Regression problem, classification on text datasets etc.

### Questions
1. Can the author suggest that this analysis still holds for strong FL algorithms like SCAFFOLD, FedDyn, pFedMe, etc.?

2. Does the authors have any theoretical justification to explain why the simple averaging of model parameters fundamentally leads to this progressive, layer-by-layer degradation in feature quality and alignment?

### Soundness
3

### Presentation
3

### Contribution
3
