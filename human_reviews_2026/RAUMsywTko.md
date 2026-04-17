# GazeVLM: Gaze-Guided Vision-Language Models for Efficient and Robust Inference

- Decision: Reject
- Scores: 2, 4, 4, 6, 4

## Abstract
Vision-language models (VLMs) are emerging as a core building block of modern intelligent assistants, enabling real-time human-machine interactions based on natural language and vision. However, the excessive number of visual tokens generated from images results in high latency, low throughput, and memory bottlenecks, which hinder real-time interactions in resource-constrained settings. To address this, we aim to reduce the number of tokens by prioritizing tokens for efficient inference using user-relevant context. With the growing usage of smart glasses, eye gaze has emerged as a promising sensing modality that can naturally convey the user intent and interests based on the user's viewing context. Therefore, it can provide useful hints for efficient inference. However, the robustness of gaze-aware VLM depends highly on the quality of gaze data. When gaze data is inaccurate, the model may overlook informative visual content, leading to degraded inference accuracy. To this end, we introduce GazeVLM, a novel gaze-guided context-aware VLM framework for efficient and robust inference under a token budget constraint. GazeVLM consists of two key phases: (i) GazeVLM-Pre: a gaze-aware preprocessing mechanism before image encoding that extracts user-attentive scenes while not losing the global understanding for robust inference; (ii) GazeVLM-Post: a gaze-guided token selection method after image encoding that prioritizes tokens around the gazing area for efficient inference under the token budget constraint. Through extensive experiments using two visual question answering datasets with real human eye-tracking data, we demonstrate that GazeVLM achieves both efficiency and robustness under varying token budgets and gaze data qualities, outperforming diverse gaze-aware and gaze-agnostic baselines. Specifically, given the budget of 500 tokens ($\approx$22\% of the tokens of the vanilla architecture), we can achieve up to 1.9$\times$ higher throughput and 37\% lower latency while slightly improving accuracy compared to the vanilla architecture.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
"GAZEVLM: GAZE-GUIDED VISION-LANGUAGE MODELS FOR EFFICIENT AND ROBUST INFERENCE" proposes a novel vision-language model (VLM) framework,  which incorporates gaze information to improve the model's inference efficiency and robustness on resource-constrained devices. Traditional VLMs typically require processing a large number of visual tokens, resulting in high latency and low throughput, making real-time interaction difficult. GAZEVLM addresses this issue through a two-stage strategy:

GAZEVLM-PRE (pre-processing stage): Extracts local and global view images based on the user's gaze point, balancing details with overall information and enhancing robustness to low-quality eye movement data.

GAZEVLM-POST (post-processing stage): After image encoding, visual tokens are prioritized based on gaze point, retaining key tokens in the gaze region while uniformly sampling tokens from other regions to meet the token budget constraint.

### Strengths
A gaze-guided token selection framework is proposed for efficient VLM inference. It selects visual tokens directly based on the user's gaze area, improving efficiency and preserving user intent.

A gaze-aware image preprocessing mechanism was designed, combining local and global views to improve the model's robustness to variations in eye movement data quality and avoid the loss of critical visual information.

A token selection strategy combining gaze and global information is proposed, which reduces the number of tokens while maintaining global understanding capabilities. It is suitable for deployment on edge devices such as smart glasses.

### Weaknesses
The paper lacks innovation in its integration of gaze and token selection. It recommends strengthening the method's uniqueness by introducing a gaze-attention fusion mechanism and comparing their complementarity.

The experiment only roughly categorizes gaze quality. Realistic noise such as offset, delay, and calibration error, is needed to validate robustness.

The paper ignores gaze bias and cross-user generalization. Leave-one-user-out testing and heatmap visualization should be conducted to diagnose whether the model over-relies on specific gaze patterns.

A static 0.5 token allocation ratio cannot adapt to content and question variations.

The model relies solely on an offline static dataset. The model needs to be deployed on edge devices such as smart glasses, and conduct user experience studies to demonstrate its efficiency and credibility in real-world scenarios.

### Questions
Please provide statistics on the overlap between gaze and the ground-truth region, as well as the accuracy drop when the IoU between gaze and ground-truth regions is less than 0.3.

The experiment only categorizes high and low quality images based on gaze offset and does not inject system noise.

Currently, a fixed β of 0.5 is used to assign gaze/surrounding tokens, but the need for local-global information varies significantly between different problems.

All experiments were conducted offline on static images, lacking end-to-end latency, energy consumption, and user subjective evaluations on real smart glasses.

The paper only reports a 1.9× throughput improvement for 500 tokens, but does not disclose the contribution of the LLM stage to total latency.

How can a circular gaze mask be aligned with the square patch boundary?

### Soundness
3

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
4

### Summary
This paper proposes GAZEVLM, a framework to improve the inference efficiency and robustness of Vision-Language Models (VLMs), particularly for real-time applications on devices like smart glasses. The core problem it addresses is the high latency and memory usage caused by processing an excessive number of visual tokens.
The proposed solution, GAZEVLM, uses eye gaze as a proxy for user intent and introduces a two-phase mechanism:
1. GAZEVLM-PRE: A gaze-aware preprocessing step that, instead of just cropping, generates two views: a "global-view" image (full scene) and a "local-view" image (gaze-centered crop). This is designed to maintain global context for robustness, inspired by foveated rendering.
2. GAZEVLM-POST: A gaze-guided token selection step after encoding. It selects a subset of tokens from both views to meet a specific token budget $T_b$. This selection prioritizes "gaze tokens" (near the gaze point) while also sampling "surrounding tokens" to retain context.
The authors evaluate GAZEVLM on two VQA datasets with real eye-tracking data (AiR-D and VQA-MHUG). The results show that GAZEVLM can achieve up to 1.9x higher throughput and 37% lower latency (with a 500-token budget, ~22% of vanilla) while slightly improving accuracy compared to the full-token baseline.

### Strengths
1. The paper proposes a method for efficient inference in VLMs while maintaining robustness, enabling their deployment on resource-constrained platforms.
2. The paper presents a novel two-phase (PRE/POST) architectural design. This design is motivated by key insights from prior work: 1) existing gaze-guided methods (e.g., pure cropping) are vulnerable to low-quality gaze data, and 2) existing efficient VLM methods (e.g., attention-based token dropping) may not align with user intent.
3. The method achieves impressive efficiency. The experiments demonstrate that with only 22% of the original tokens, the architecture can achieve higher accuracy, 1.9x higher throughput, and maintain robustness.

### Weaknesses
1. The novelty of the individual components is limited. GAZEVLM-PRE is essentially a combination of "center cropping + global scaling," and GAZEVLM-POST combines "Euclidean distance-based selection + uniform sampling." These techniques, in isolation, are not new.

2. The performance of the baselines seems questionable. The HiRED baseline performs exceptionally poorly (61.4% accuracy), and the paper's justification (misalignment with user intent) is a strong claim that may not be fully supported. An unfair implementation is a possible alternative explanation. More importantly, the GAZEVLM-PRE ablation (1,200 tokens) already outperforms the Vanilla baseline (2,295 tokens) in both accuracy (76.4% vs. 74.3%) and throughput (2.02 vs. 1.42). This implies that the 2-view foveated partitioning (PRE) might be the main source of improvement, rather than the POST token selection. The paper attributes the gains to the combined PRE+POST framework, but the individual contributions are not clearly disentangled.

3. There is a significant mismatch between the motivation and the experimental setup. The paper motivates the work with real-time, streaming applications (XR, VR, autonomous driving), but the evaluation is conducted on static VQA datasets. These datasets are limited, and the methodology of using only the "last gaze point" is an oversimplification that seems arbitrary and is not well-justified.

### Questions
1. Please explain the discrepancy between the 2,880 tokens for the "Vanilla" baseline mentioned in the text (Footnote 1, Appendix A.2) and the 2,295 tokens used in all experimental tables (Table 1, 3, 4)? Which number is correct, and how are the efficiency gains calculated?
2. The definition of "gaze deviation" used for the robustness study (Fig 4) is unclear. The split ("top 10% low deviation" vs. "the rest") feels arbitrary. How was "gaze deviation" calculated? What was the gaze deviating from? Would you consider a more systematic robustness evaluation by simulating noise on the gaze coordinates?
3. The GAZEVLM-PRE ablation (Table 3) outperforms the Vanilla baseline in both accuracy and efficiency. Does this imply that the LLaVA-Next 5-view partitioning is simply a poor design and that your 2-view foveated partitioning is the main source of improvement, rather than the GAZEVLM-POST token selection? Is the comparison between a 5-partition and 2-partition architecture a fair baseline?
4. The data processing uses only the last gaze point, which is a significant simplification. What is the justification for this? Why not use the point with the longest fixation duration, or the centroid of the entire scanpath?
5. The HiRED baseline performs very poorly. Could you please clarify its implementation?
6. Can the practical significance of the hyperparameters $\alpha$ and $\beta$ be explained beyond a simple parameter sweep? How do these two parameters interact, and do they determine how the PRE and POST modules collaborate?

### Soundness
3

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
3

### Summary
This paper introduces GazeVLM, a novel framework that leverages eye gaze data to improve the efficiency and robustness of Vision-Language Model (VLM) inference. The approach consists of two key components: (1) GazeVLM-PRE: a gaze-aware preprocessing mechanism that extracts both local-view (gaze-focused) and global-view images before encoding, and (2) GazeVLM-POST: a gaze-guided token selection method after encoding that prioritizes tokens around the gazing area while maintaining global context. The authors demonstrate that their approach achieves higher throughput and lower latency while using only 22% of tokens compared to vanilla architectures, with minimal or even improved accuracy on two VQA datasets with real eye-tracking data.

### Strengths
1. The paper addresses a relevant problem for deploying VLMs on resource-constrained smart glasses and XR devices, with clear motivation for using eye gaze as a user intent signal.
2. The combination of preprocessing (for robustness) and postprocessing (for budget control) is intuitive and shows practical benefits.
3. Unlike existing cropping-based approaches, the paper explicitly considers gaze quality variations and demonstrates maintained performance under low-quality gaze data.

### Weaknesses
1. Lack of Theoretical Foundation and Principled Design. The paper is primarily engineering-driven with limited theoretical justification for its design choices. Critical parameters such as α = 0.5 (gaze region ratio) and β = 0.5 (gaze token ratio) are selected purely through empirical grid search without principled reasoning. The even split between local/global views (T_v = ½T_a) and the choice of circular regions for gaze tokens lack any theoretical or optimization-based justification. More fundamentally, the paper provides no information-theoretic analysis of why gaze-guided selection should preserve task-relevant information, no formal characterization of the trade-off between local detail and global context, and no principled framework for when gaze guidance helps versus hurts performance. The token selection strategy based on simple Euclidean distance and uniform sampling is overly simplistic, as it ignores semantic relationships between tokens, doesn't consider token importance beyond spatial proximity, and doesn't explore learning-based or attention-weighted selection strategies. 

2. Severely Limited Experimental Diversity and Scope. The experimental validation relies on only 2 datasets (AiR-D and VQA-MHUG), both collected in controlled settings rather than real-world scenarios, both relatively small (10K and 8K samples), and both limited to VQA tasks only. There is no evaluation on other critical vision-language tasks such as image captioning, visual grounding, visual reasoning, or video understanding, all of which are essential for smart glasses applications. The model coverage is equally narrow, testing only LLaVA variants (Mistral-7B, Llama3-8B) without exploring other VLM architectures or larger models where efficiency gains would be more impactful. The paper also lacks evaluation on scenarios where gaze might be ambiguous (multiple similar objects), dynamic scenes, multi-turn dialogues, or tasks requiring global reasoning (counting, spatial relationships). This limited scope raises serious concerns about the generalizability of the findings and whether the method works beyond the specific controlled settings tested.

3. Overly Simplistic and Unjustified Gaze Data Utilization. The paper's use of gaze data discards rich information that could improve performance. Using only the "last gaze point" is never justified and ignores valuable temporal patterns—scanpath sequences reveal cognitive processes, fixation durations indicate importance levels, and first versus later fixations have different semantic meanings. The binary quality split (top 10% versus rest) is arbitrary with no justification, missing opportunities for continuous modeling of gaze uncertainty or adaptive strategies based on confidence scores. The paper doesn't leverage gaze prediction uncertainty information that real eye trackers provide, doesn't employ probabilistic modeling of gaze locations, and doesn't handle gaze prediction errors in a principled way. Furthermore, there's no ablation study comparing different gaze extraction strategies (first fixation, longest fixation, centroid, duration-weighted average), making it impossible to know if the chosen approach is optimal or even reasonable.

4. Incomplete Baseline Comparisons and Missing State-of-the-Art Methods. The paper fails to compare against several recent and relevant efficient VLM methods, including LLaVA-PruMerge (which is mentioned in related work but never compared), FastV, VisionZip, VideoLLM-online, VideoLLM-MoD, and MiniCache. The comparison with HiRED appears somewhat unfair since HiRED is designed for general visual importance rather than user-specific context, yet the paper doesn't discuss scenarios where visual importance might align with user intent or explore hybrid approaches combining both signals. There's also no comparison with learned token selection methods or reinforcement learning-based approaches, which could potentially outperform the heuristic-based selection. This incomplete comparison makes it difficult to assess whether the performance gains come from gaze integration specifically or simply from having any reasonable token selection strategy, and whether more sophisticated methods could achieve better results.

5. Insufficient Analysis, Ablations, and Methodological Rigor. The paper lacks critical analyses that would provide insights into when and why the method works. There's no failure case analysis explaining when GazeVLM fails, no breakdown by question types (spatial vs. semantic), no analysis of how performance varies with image complexity, and no reporting of computational overhead for preprocessing. The ablation studies are limited—there's no exploration of different token selection strategies (circular vs. elliptical vs. attention-weighted regions), no comparison of gaze extraction methods, and no study on the optimal number of gaze points to use. Methodologically, the paper only reports accuracy without other important metrics (F1, precision-recall for specific object types), lacks human evaluation of response quality, provides no confidence intervals or significance tests for the reported improvements, and doesn't specify important details like the "deviation" metric for gaze quality assessment or random seeds for reproducibility.

### Questions
1. Can you provide information-theoretic or optimization-theoretic justification for your design choices? Why should gaze-guided selection preserve task-relevant information?
2. Why not learn α and β from data? Why not make them adaptive based on gaze quality, image complexity, or question type?
3. Have you compared different gaze extraction strategies (first fixation, longest fixation, centroid, weighted average by duration)? Why is "last gaze point" optimal?
4. How does your method perform on tasks requiring global reasoning (counting, spatial relationships)?
5. Why not compare with recent efficient VLM methods like LLaVA-PruMerge, FastV, or VisionZip? How does your method compare to learned token selection?
6. In what scenarios does gaze guidance hurt performance? When should we prefer gaze-agnostic methods?

### Soundness
3

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
5

### Summary
The authors present a novel training-free efficient VLM inference framework called GazeVLM. GazeVLM utilizes gaze information for a visual stimulus to ascertain important tokens that must be utilized by the VLM to respond to a textual prompt. This allows the VLM to operate under strict token budget constraints, thereby gaining throughput and reducing memory footprint, while not sacrificing performance. Through experiments and analysis, the authors claim that GazeVLM is able to outperform previous methods in terms of efficacy and efficiency.

### Strengths
(1) Great motivation: I find the core intuition of using gaze to make VLMs more token-efficient in terms of the vision modality relevant and useful. Making VLMs more context-aware, i.e., understand user’s visual attention, while participating in conversation with the user is also a task worth studying.

(2) Adequate model design: I find the way the authors have analyzed the limitations of existing efficient VLMs and proposed a solution to mitigate these limitations worthy of appreciation. The training-free approach is both straightforward and effective.

(3) Thorough experimental analysis: The authors have performed a very thorough and comprehensive set of experiments and ablations in both the main text and the appendix.

### Weaknesses
(1) Method might be proved redundant by better vision-language alignment: Even though the solution is novel and well-motivated, as VLMs get more sophisticated (as shown in Appendix A.3), e.g., learn to choose visual tokens that are more relevant to the prompt, explicitly using gaze might lose its efficacy. In an early study, it was revealed that between 70-95% of  fixated objects are described in the corresponding language descriptions. In that case, learning better vision-language alignment might be sufficient to deduce useful visual tokens. 

(2) Performance gains are not significant: Simple baselines like Pooling (2 X 2) and Dotted Map are not significantly outperformed by GazeVLM in efficiency/efficacy metrics. I wonder for other simpler tasks, like object localization, if these simple baselines may fare against GazeVLM. From Figure 4, Circled Map does better than GazeVLM for high quality gaze. However, we expect eye trackers to get more sophisticated, and expect gaze data to be better quality as days go by.

### Questions
(1) What happens if you don’t provide the global-view tokens and force GazeVLM to rely only on local-view image?

(2) Any strong reason for choosing not to fine-tune/retrain a VLM like Voila-A but using the core intuition of the paper?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper integrates gaze into VLM inference with a two-stage approach, GAZEVLM-Pre and GAZEVLM-Post. In GAZEVLM-Pre, the authors create two views of each image: a global full-scene view and a local view cropped around the gaze point. In GAZEVLM-Post, they apply token-selection strategies to keep only a subset of vision tokens, reducing the token budget to improve efficiency and enable deployment on wearable devices. They compare their approach against several baselines on two VQA datasets with real human eye-tracking (AiR-D and VQA-MHUG).

### Strengths
The paper tackles a real problem in modern VLMs: high inference cost that prevents practical use in interactive AR/smart-glasses. Reducing visual tokens while preserving accuracy is a sensible way forward. The authors propose a simple, effective method that lowers memory usage and increases throughput without sacrificing accuracy.

### Weaknesses
The paper could be clearer. There’s repeated text between the background and introduction, and the method section needs more plain, consistent notation. The post-gaze stage is under-explained: it’s not clear how tokens are picked, what rules guide the choice, or how the token budget is enforced. It’s also not obvious why a simple gaze-centered crop wouldn’t work just as well. Both approaches need tuning (token budget/ratios vs. crop size), so a direct, well-tuned comparison to a cropping baseline would help show the real benefit.

The evaluation is limited to two lab-style datasets. Results on truly egocentric, in-the-wild data would make the claims stronger—e.g., testing on Ego4D (if suitable gaze annotations are available).

### Questions
1. Could you run a controlled study where you inject noise into the gaze coordinates and plot accuracy vs. noise level for GAZEVLM and the baselines?

2. In POST, you use a fixed 50/50 split between gaze-near tokens and uniformly sampled “surrounding” tokens. Is that always best? Did you try learning this allocation or adapting it by question type (e.g., “What color is the sign I’m looking at?” might want mostly gaze tokens, while “Where are we?” needs more global tokens). Also, do you have stats on how token sources vary by image/question type?

3. How do you avoid redundant/duplicate visual tokens between the global and local views? Since the global view already contains the local region, why encode both views and then select tokens from each, instead of encoding only the global image and dropping tokens far from the gaze center?

4. Could a global-only + gaze-aware token selection baseline match your accuracy/latency without the second (local) pass? If not, can you show an ablation?

5 . All experiments are on an A100-80GB with batch size 1, which is a very strong setup. For the smart-glasses story, what on-device or edge hardware are you actually targeting, and is the 300–500 token regime small enough for real-time on those devices (latency, memory, and power)? If possible, can you share end-to-end numbers on a representative edge platform?

### Soundness
2

### Presentation
2

### Contribution
2
