# Where is Motion From? Scalable Motion Attribution for Video Generation Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 8, 2, 4

## Abstract
Despite the rapid progress of video generative models, the role of data in shaping motion quality is poorly understood. We present MOTIVE (MOtion Training Influence for Video gEneration), a motion-centric, gradient-based data attribution framework that scales to modern, large, high-quality video datasets and models. We use this to study which finetuning clips improve or degrade temporal dynamics. MOTIVE isolates temporal dynamics from static appearance via flow-weighted loss masks, yielding scalable influence scores practical for modern, large, and high-quality datasets and models. On text-to-video models, MOTIVE identifies clips that strongly affect motion and guides data curation that improves temporal consistency and physical plausibility. With MOTIVE selected high-influence data, our method improves both motion smoothness and dynamic degree on VBench, achieving a 74.1% human preference win rate compared with the pretrained base model. To our knowledge, this is the first framework that attributes motion (not just appearance) in video generative models and uses it to curate finetuning data.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes selecting a subset of motion-influential data for training video generative models. The goal is to choose data that strongly affects motion and to guide data curation in a way that improves temporal consistency and physical plausibility. A computational method is used to compare the gradients of query and training data points for this purpose.

### Strengths
1. The paper is well-written and easy to follow.

2. Both computational experiments and human studies are provided to support the claims.

### Weaknesses
**Scalability concern:**  In the experiment, there are 50 query videos and 10k clips in the training subset. Adding just one more query requires computing influence scores 10k additional times. How to address this limit?

**Equation (5) issue:** How can the exact cardinality of the sampled timestep-and-noise set \\( \\mathcal{T} \\) be computed? There are infinitely many choices for the noise vector and time step in the flow matching framework. Shouldn’t it be better represented as an expectation?  

**Query data instance:** Can we use a self-generated video and what does it imply?

**Equation (7) parameter:**   fixing $\\epsilon_{\\text{fix}}$ is sort of understandable. But why is the timestep $t_{\\text{fix}}$ fixed?  

**Equation (8) – Structured projection:** Why select the structured projection in this way? Many terms involve random operators; why is a statistical operator (e.g., expectation) not needed?  

**Comparison metrics:**  Suggest using other metrics such as FVMD for more comprehensive quantitative results [1].  

**Human evaluation experiment:**  Only 20 videos are used (10 from a baseline, 10 from the proposed method).  This seems too few to yield convincing results.


Ref:\
[1] Liu, J., Qu, Y., Yan, Q., Zeng, X., Wang, L. and Liao, R., 2024. Fr\'echet Video Motion Distance: A Metric for Evaluating Motion Consistency in Videos. arXiv preprint arXiv:2407.16124.

### Questions
Please see the [Weakness] section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper addresses the underexplored issue of motion attribution in video generative models by proposing MOTIVE, a scalable gradient-based framework. Existing methods fail to separate motion from static appearance and lack scalability for large datasets/models. MOTIVE tackles this via three key steps: using flow-weighted masks to isolate temporal dynamics, correcting frame-length bias for fair scoring, and applying Fastfood projection for efficient gradient storage/computation. It calculates motion influence scores for training clips, selects top 10% high-impact data for finetuning. Experiments on VIDGEN-1M/4DNeX-10M with Wan2.1-T2V-1.3B show MOTIVE outperforms baselines: 89.4% dynamic degree on VBench (surpassing full-dataset finetuning’s 84.7%) and 76.7% human preference win rate vs. pretrained models, proving its value for targeted data curation and motion quality improvement.

### Strengths
1. Critical and Well-Targeted Problem FormulationThe paper focuses on a pivotal, underexplored gap in video generative models: identifying training data that drives motion quality (a core video feature distinct from static appearance). Filtering high-quality motion data is vital for finetuning—where carefully selected clips significantly boost temporal coherence and physical plausibility—filling a key need for practical video model optimization.

2. Intuitive and Principled Method DesignMOTIVE’s approach is highly logical and video-specific. It isolates motion from static content via flow-weighted loss masks (using optical flow for dynamic regions), corrects frame-length bias (avoiding spurious long-clip ranking), and uses Fastfood projection for scalability. These choices directly fix image-centric attribution limits, making the framework theoretically sound and practically feasible.

3. Comprehensive and Rigorous ExperimentsExperiments are thorough: evaluations on VIDGEN-1M/4DNeX-10M, VBench for motion metrics, diverse baselines (random selection, full finetuning), and human evaluations (76.7% preference vs. base model). Ablations (single-timestep validity, projection dimension impact) validate components, ensuring robust, credible conclusions.

### Weaknesses
All experiments in the paper are exclusively conducted on the Wan2.1-T2V-1.3B model, with no validation on other mainstream video generative architectures (e.g., 3D U-Nets, latent video VAEs with different temporal attention blocks, or non-DiT-based diffusion models). As the paper itself acknowledges, "our evaluation centers on one open-source backbone due to compute; broader portability is future work" — this single-model focus means the framework’s effectiveness, such as motion mask compatibility, gradient projection stability, and finetuning gain consistency, cannot be confirmed for other popular video diffusion models. This limitation reduces confidence in the framework’s general applicability to diverse video generation systems, weakening the persuasiveness of its universal utility.

### Questions
This paper addresses a crucial problem in data of video diffusion training with detailed analysis, making it worthy of acceptance. As it only tests Wan2.1-T2V-1.3B, adding one more model test would further boost its quality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the under-explored problem of attributing motion in generated videos to specific training clips, introducing MOTIVE, a gradient-based, motion-aware data attribution framework for video diffusion models. The key idea is to isolate temporal dynamics from static appearance by re-weighting gradients with flow-derived motion masks, enabling scalable influence estimation over modern billion-parameter models. Extensive experiments on VIDGEN-1M and 4DNeX-10M show great performance of MOTIVE.

### Strengths
- This paper opens an important new topic—motion-centric data attribution—that prior image-oriented methods cannot address.

- The authors propose a simple yet effective idea: flow-weighted gradient masking that disentangles motion from appearance without changing the forward generative process.

- Experiments are good, including large-scale datasets and human evaluations.

### Weaknesses
I would like to begin by setting aside the specific technical details and share my perspective on the problem that this paper aims to address. Based on my research experience in video generation, I have observed a clear trade-off between motion dynamics and the occurrence of visual artifacts—in general, stronger motion dynamics tend to correlate with a higher probability of artifacts. I believe this trade-off represents one of the most fundamental challenges in current video generation research. From the experimental results presented in the paper (e.g., Table 1), it appears that the proposed method still suffers from this dilemma. In other words, while the approach seems to improve the Dynamic Degree metric significantly, this improvement may come at the cost of other important aspects such as Background Consistency and Imaging Quality. If the main contribution of this work is limited to enhancing motion dynamics at the expense of overall visual stability and quality, such improvements could arguably be achieved more simply by using training data with inherently higher motion. Overall, I appreciate the authors’ motivation to address the problem of video dynamism. However, the current method seems not to fundamentally resolve the underlying trade-off and not to tackle the core technical challenges of dynamic yet consistent video generation.

In addition, I believe the paper has the following shortcomings:
(1) Writing quality—the overall writing could be improved. For example, each equation should be properly punctuated, and the presentation could benefit from more polished academic writing.
(2) Experimental sufficiency—the experiments are relatively limited, as they are conducted only on the Wan-2.1 model. This raises concerns about the reliability and generalizability of the results. It would be more convincing if the authors could include additional models, such as HunyuanVideo, to validate the effectiveness and robustness of their approach.

### Questions
see weakness

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper introduces MOTIVE, a framework for motion-centric data attribution in video diffusion models. While existing attribution methods primarily analyze static appearance in image diffusion, MOTIVE aims to identify which finetuning clips most influence temporal dynamics in generated videos. The key idea is to compute motion-weighted gradients, where optical-flow–based masks emphasize dynamic regions while suppressing static backgrounds.

### Strengths
The paper identifies an underexplored but meaningful question: “Which training data drives motion learning in video diffusion models?” This focus on motion attribution provides a clear conceptual step beyond standard image-level attribution, and the method effectively extends existing data attribution frameworks to the video domain with appropriate modifications for temporal structure and scalability.

### Weaknesses
**Limited evaluation and unclear attribution advantage**  
The proposed attribution-based selection may in practice act as a proxy for identifying motion-rich or high-quality clips, rather than truly capturing data that causally influences motion learning. This concern is amplified by the evaluation setup, where the method is compared only against random selection, a trivial baseline that cannot disentangle whether improvements stem from genuine attribution or simply from favoring dynamic, well-captured videos. A more meaningful comparison would involve finetuning with datasets selected by explicit motion-quality criteria, such as average motion magnitude, optical-flow statistics, or reward-model scores reflecting motion realism or physical plausibility. Without such baselines, it remains unclear whether the proposed approach provides any advantage beyond straightforward motion-saliency filtering.

### Questions
See the weakness.

### Soundness
3

### Presentation
3

### Contribution
2
