# D2E: Scaling Vision-Action Pretraining on Desktop Data for Transfer to Embodied AI

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 8

## Abstract
Large language models leverage internet-scale text data, yet embodied AI remains constrained by the prohibitive costs of physical trajectory collection.
Desktop environments---particularly gaming---offer a compelling alternative: they provide rich sensorimotor interactions at scale while maintaining the structured observation-action coupling essential for embodied learning.
We present D2E (Desktop to Embodied AI), a framework that demonstrates desktop interactions can serve as an effective pretraining substrate for robotics embodied AI tasks.
Unlike prior work that remained domain-specific (e.g., VPT for Minecraft) or kept data proprietary (e.g., SIMA), D2E establishes a complete pipeline from scalable desktop data collection to verified transfer in embodied domains.
Our framework comprises three components: (1) the OWA Toolkit that unifies diverse desktop interactions into a standardized format with 152× compression, (2) the Generalist-IDM that achieves strong zero-shot generalization across unseen games through timestamp-based event prediction, enabling internet-scale pseudo-labeling, and (3) VAPT that transfers desktop-pretrained representations to physical manipulation and navigation.
Using 1.3K+ hours of data (259 hours of human demonstrations and 1K+ hours of pseudo-labeled gameplay), our 1B-parameter model achieves 96.6\% success on LIBERO manipulation and 83.3\% on CANVAS navigation, matching or surpassing models up to 7$\times$ larger, such as $\pi_0$ (3.3B) and OpenVLA (7B).
These results demonstrate that sensorimotor primitives learned from digital interactions transfer effectively to real-world physical tasks, establishing desktop pretraining as a practical paradigm for embodied AI.
All resources are publicly available at https://worv-ai.github.io/d2e.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes D2E (Desktop-to-Embodied AI), a framework that leverages large-scale desktop interaction data (screen, keyboard, mouse) as an alternative to expensive real-world embodied trajectories for pretraining vision-action models.
The system consists of three main components: (1) OWA Toolkit – a high-performance desktop data collection and compression pipeline based on an extended MCAP format (OWAMcap), achieving up to 152× compression over prior datasets. (2)Generalist-IDM – a timestamp-aware inverse dynamics model (NEP-τ) that predicts human actions from videos and is used to pseudo-label over 1K hours of YouTube gameplay. (3)VAPT – a vision-action pretrained model that transfers desktop-learned representations to robot manipulation (LIBERO) and navigation (CANVAS), achieving 96.6% and 83.3% success rates respectively.

### Strengths
1. **Solid engineering contribution**: The OWA Toolkit is an impressive system-level effort that enables synchronized, multimodal desktop data collection and efficient storage. The compression performance and open-source reproducibility are highly commendable.

2. **Reproducibility and openness**: The authors provide thorough implementation details, training settings, and datasets, which greatly enhance the paper’s credibility and community value.

### Weaknesses
1. **Lack of academic novelty**:
The overall contribution is primarily engineering-oriented rather than conceptual or algorithmic. While the proposed OWA Toolkit and data infrastructure are impressive from a systems perspective, the work introduces limited new ideas in terms of representation learning, model design, or theoretical insight.

2. **Generalist-IDM design appears incremental**:
The core methodological component—Generalist Inverse Dynamics Model (Generalist-IDM)—is not sufficiently novel or deeply motivated. The NEP-τ formulation is essentially an incremental extension of standard next-event prediction, and its role in improving downstream performance remains unclear. It is uncertain whether the strong results of VAPT stem from the NEP-τ design itself or simply from the high-quality and diverse data collected by OWA Toolkit.

3. **Lack of ablation and sensitivity analysis**:
The paper provides limited empirical evidence dissecting the effectiveness of individual components. In particular, the temporal offset parameter τ is introduced as a key idea, yet there is no systematic study on how τ is selected or how sensitive the model performance is to its value. Without such analysis, the robustness and generality of the proposed modeling choice remain uncertain.

4. **Missing causal connection between design and outcome**:
Although the experiments show that a model trained on limited human desktop data and pseudo-labeled gameplay videos can generalize across unseen domains, the paper does not clearly explain why this happens. It lacks causal analysis connecting the proposed components (e.g., NEP-τ, OWA data quality, pseudo-labeling) to the observed generalization behavior. As a result, it is difficult to attribute the performance gains to specific methodological factors rather than data scale or diversity alone.

### Questions
1. Could the authors provide a deeper analysis of how the temporal offset mechanism (NEP-τ) contributes to the observed improvements? In particular, how do results change if τ is removed or varied?

2. To what extent do the downstream results of VAPT originate from the model design (NEP-τ, Generalist-IDM) versus the quality and diversity of the collected OWA data? Some controlled comparisons would help isolate these factors.

3. The paper shows strong zero-shot transfer to unseen games and robotic tasks. What is the hypothesized mechanism behind this generalization? Are certain components (e.g., timestamp-based tokenization) more critical than others?

### Soundness
2

### Presentation
2

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
The paper introduces D2E, a framework that leverages large-scale desktop interactions collected via the OWA toolkit and generalized through a timestamp-based IDM for pretraining embodied AI. Using more than 1.3K hours of human and pseudo-labeled gameplay data, the approach demonstrates strong transfer to robotics tasks, achieving 96.6% on LIBERO manipulation and 83.3% on CANVAS navigation.

### Strengths
1. Using game data for embodied pretraining is an interesting direction.

2. The paper provides a detailed system design for data collection.

### Weaknesses
1. The main contribution of the paper lies in how to collect action data and in the system-level design, while the algorithmic innovation is relatively limited. Currently, there is a large body of work that uses OOD data for pretraining, so my concern is that this paper may not be a very good fit for a learning-focused conference.

2. Using game data for navigation tasks in CANVAS makes sense, but its effectiveness for manipulation tasks remains questionable. In addition, LIBERO can be easily hacked with certain tricks, which makes it difficult to serve as a fair or reliable benchmark for validating algorithms.

### Questions
Did the pretraining stage also include robotics data, or was it purely based on desktop game data?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper presents D2E, a framework that scales vision-action pretraining using large-scale desktop interaction data (screen, keyboard, mouse) instead of costly physical robot trajectories. It introduces the OWA Toolkit for efficient multimodal data capture, an IDM for pseudo-labeling internet gameplay videos, and a VAPT that transfers learned representations to robotics. Trained on about 1.3K hours of human and pseudo-labeled data, the model achieves 96.6% success on libero and 83.3% on canvas navigation, showing that desktop interactions can effectively serve as scalable pretraining for embodied AI.

### Strengths
1. End-to-end pipeline and scale: The paper provides a complete framework covering data collection, pseudo-labeling, pretraining, and robotic transfer. It builds on 31 desktop games with 335 hours of human demonstrations and expands to over 1,000 hours of pseudo-labeled YouTube gameplay, achieving clear scalability.

2. Solid engineering contribution: The OWA toolkit enables synchronized screen, keyboard, and mouse capture with nanosecond precision and 152× compression, significantly reducing random-access I/O cost and improving data-loading throughput during training.

3. Openness and reproducibility: The authors commit to releasing the toolkit, datasets, and pretrained models with detailed documentation, ensuring transparency and easy reproducibility for future research.

### Weaknesses
The experimental evidence supporting why and when desktop data benefits embodied tasks remains insufficient. While results show positive transfer, the paper lacks a deeper analysis of task suitability—for example, whether desktop-derived sensorimotor patterns genuinely align with the fine-grained control and contact dynamics required in manipulation tasks.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes D2E, a framework for pretraining models on desktop data (e.g., game videos paired with human actions). D2E offers a comprehensive toolbox that includes standardized data collection and storage, efficient data reading, and a scalable pseudo-label annotation strategy for desktop data, powered by a specifically designed and trained Inverse Dynamics Model (IDM). Extensive experiments are conducted to validate the framework across multiple aspects, including data collection, processing, and reading efficiency, as well as the effectiveness of using D2E for pretraining embodied models. The results show that adapting the pretrained models to embodied and navigation tasks leads to notable performance improvements, revealing a promising direction for leveraging scalable desktop data in embodied model pretraining.

### Strengths
1. Given the high cost of embodied data collection, exploring new data sources that capture human intention knowledge and can contribute to embodied learning is a highly meaningful research direction. While desktop data has been explored in previous works for training embodied models, those methods primarily focus on developing and validating models within in-domain game environments. This paper extends beyond that scope, aiming to make desktop data more broadly useful for embodied learning.

2. The high-quality datasets collected and the well-designed data collection pipeline represent valuable contributions to the research community. They provide essential infrastructure for future work on large-scale embodied pretraining, facilitating both reproducibility and scalability.

3. The experiments and analyses conducted on each component of the proposed framework are comprehensive. Beyond demonstrating the feasibility of pretraining on desktop data, the paper offers a practical reference and a solid foundation for developing future learning systems capable of utilizing any video-based data for embodied intelligence.

4. The significant performance gains observed on downstream tasks following D2E pretraining clearly demonstrate the framework’s effectiveness in extracting generalizable knowledge from desktop data, highlighting its potential as a scalable and efficient pretraining paradigm.

### Weaknesses
1. The utilization of data with machine-generated pseudo labels leads to inconsistent performance changes across the Manipulation and Navigation benchmarks. This inconsistency suggests that the quality and reliability of pseudo labels may vary significantly depending on the task type, and further analysis is needed to understand their impact on downstream performance.

2. Conducting experiments solely on the Libero benchmark is insufficient to support the claim that pretraining on desktop data contributes to learning generalized embodied knowledge. Validation on real-world manipulation tasks would substantially strengthen the paper’s argument. 

## Minor Issue

1. The font style used in the paper is inconsistent with the official ICLR template, and should be adjusted to comply with the formatting standards.

### Questions
1. Could the authors provide more details about the baseline implementations reported in Table 1? It would be helpful to clarify whether these baselines were reimplemented under the same settings or adopted from existing works.

2. There remains a considerable gap between game data and embodied manipulation data due to the coarse-grained differences in action spaces and the varying requirements for 3D physical understanding. A more comprehensive analysis is needed to quantify how much embodied manipulation tasks actually benefit from desktop data pretraining, and under what conditions such transfer is most effective.

### Soundness
3

### Presentation
3

### Contribution
3
