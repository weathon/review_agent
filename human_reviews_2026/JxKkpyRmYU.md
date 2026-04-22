# C2-Evo: Co-Evolving Multimodal Data and Model for  Self-Improving Reasoning

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 4, 6

## Abstract
Recent advances in multimodal large language models (MLLMs) have shown impressive reasoning capabilities. However, further enhancing existing MLLMs necessitates high-quality vision-language datasets with carefully curated task complexities, which are both costly and challenging to scale. Although recent self-improving models that iteratively refine themselves offer a feasible solution, they still suffer from two core challenges: (i) most existing methods augment visual or textual data separately, resulting in discrepancies in data complexity (e.g., over-simplified diagrams paired with redundant textual descriptions); and (ii) the evolution of data and models is also separated, leading to scenarios where models are exposed to tasks with mismatched difficulty levels. To address these issues, we propose C2-Evo, an automatic, closed-loop self-improving framework that jointly evolves both training data and model capabilities. Specifically, given a base dataset and a base model, C2-Evo enhances them by a cross-modal data evolution loop and a data-model evolution loop. The former loop expands the base dataset by generating complex multimodal problems that combine structured textual sub-problems with iteratively specified geometric diagrams or mathematical functions, while the latter loop adaptively selects the generated problems based on the performance of the base model, to conduct supervised fine-tuning and reinforcement learning alternately. Consequently, our method continuously refines its model and training data, and consistently obtains considerable performance gains across multiple mathematical reasoning benchmarks. Our code, models, and datasets will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes C2-Evo, a closed-loop framework that jointly evolves training data and model capabilities. C2-Evo integrates cross-modal data evolution with adaptive model training. Experiments on several mathematical reasoning benchmarks show that C2-Evo yields considerable performance gains without external data.

### Strengths
• This paper is well-organized and easy to read.

• Developing self-improving reasoning capabilities in MLLMs is an important and interesting task. 

• The proposed C2-Evo is reasonable and can be applied to various fine-tuning scenarios.

### Weaknesses
• The proposed method is incremental, combining data-evolving and model-evolving methods. In particular, the co-evolving data and model method merely applies widely adopted SFT and GRPO with accuracy reward and format reward to fine-tune the MLLMs. 

• It would be better to include some advanced data-evolving and method-evolving baselines. It would be unfair to compare the original LLMs (Qwen2-VL).

### Questions
It would be better to include some advanced data-evolving and method-evolving baselines. It would be unfair to compare the original LLMs (Qwen2-VL).

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces **C2-Evo**, a self-improving framework designed to jointly enhance multimodal large language models (MLLMs) and their training data. The framework addresses two key challenges in existing methods: mismatched visual-textual data complexities and the separation of data evolution from model improvement.  

C2-Evo operates through two integrated loops: (i) a cross-modal data evolution loop that generates complex multimodal problems with balanced challenges, and (ii) a data-model evolution loop that adapts training tasks based on model performance. Through these continuous iterations, C2-Evo achieves notable improvements across mathematical reasoning benchmarks.

### Strengths
1. **Well-Presented and Clear**: The paper is written in a clear and accessible manner, ensuring that readers can easily understand, reproduce, and validate the proposed methods.  

2. **Unified Evolution of Vision and Text**: The paper addresses the critical challenge of integrating the evolution of visual and textual data into a unified process, allowing simultaneous evolution of images and instruction texts. This significantly enhances the diversity of the dataset. As image evolution is inherently challenging, the authors wisely focus on subcategories like mathematical multimodal question answering, which are easier to implement via code modifications while highlighting the core contributions effectively.  

3. **Use of SFT and Advanced Post-Training Techniques**: In addition to supervised fine-tuning (SFT), the paper employs GRPO and other post-training techniques to improve model performance in targeted domains such as mathematics. This not only aligns with current research trends but also helps deliver further performance enhancements in these specialized subfields.

### Weaknesses
1. **Overclaiming the Contribution of Cyclic Evolution**: While the paper introduces the concept of cyclic evolution between data and models under the name C2-Evo, this approach is already common in prior works [1,2]. The lack of discussion around these related classic studies may lead to an overstatement of its contribution and novelty.  

2. **Use of Outdated Base Models**: The experiments are conducted using Qwen2 series models, despite the availability of the newer Qwen3 series models. Results obtained using older models may not be fully convincing. It is necessary to validate the method with at least the Qwen2.5 series, which could better assess the timeliness and adaptability of the proposed data evolution mechanism.  

3. **Challenges in Image Evolution and Its Necessity**: While image evolution is acknowledged as a difficult task, even with narrowed subcategories such as mathematics, the diversity of evolved images may still be limited, and there is a risk of introducing errors. This raises doubts about the necessity of performing image evolution. Ablation studies focusing solely on evolving textual instructions (without image synchronization) are necessary to provide evidence for the added value of image evolution, but the paper does not include such experiments.

[1]  MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct

[2]  VILA^2: VILA Augmented VILA

### Questions
1. **Similarity to Existing Works Without Explicit Comparison**: The presentation style of this work, including the shift in reasoning step distribution and the complexity analysis of instruction data, closely resembles the MMEvol [1] study. However, the authors have not directly compared their method to MMEvol or clarified the distinctive contributions of their approach. A detailed classification and differentiation of their core contributions are needed to establish novelty.  

2. **Questionable Necessity and Generality of Image Evolution**: While the paper focuses on multimodal evolution, it fails to prove the necessity of image evolution. Furthermore, no experiments were conducted on general multimodal datasets, raising doubts about the generality and utility of the approach. When scaled to a larger dataset size, such as 10M samples for large-scale training, the evolved data may struggle to significantly enhance multimodal model performance. For instance, while MMEvol shows notable improvement on small datasets, its impact diminishes significantly when scaled to 10M data points (e.g., Mammthvl [2]). This could suggest a similar scalability limitation for C2-Evo.  

3. **Ambiguous Claim of "Self-Improving"**: The paper casually claims to achieve self-improving capabilities, but it still relies heavily on advanced multimodal models like GPT-4o for assistance, making this more akin to a form of data distillation rather than true self-improvement. Methods like ViLA^2 [3], which more rigorously align with self-improving frameworks, might be more fitting representatives of this paradigm. Further clarification and justification of C2-Evo's self-improvement claim are necessary.

[1]  MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct

[2] MAmmoTH-VL: Eliciting Multimodal Reasoning with Instruction Tuning at Scale

[3]  VILA^2: VILA Augmented VILA

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
4

### Summary
The paper proposes C2-Evo, a self-improving multimodal reasoning framework designed to address two key challenges in current Multimodal Large Language Models (MLLMs): (1) the mismatch between visual and textual data complexity, and (2) the misalignment between model capability and task difficulty. C2-Evo introduces two closed-loop co-evolution mechanisms: a cross-modal data co evolution loop, which generates semantically consistent and progressively more complex visual-textual reasoning problems (e.g., geometric diagrams, mathematical functions) using GPT-4o; and a data–model co-evolution loop, which dynamically selects training samples based on model error rates and alternates between Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO). Experiments on benchmarks such as Geo-Sub, MathVista, and MathVerse show consistent improvement across three iterations. Notably, C2-Evo achieves performance close to GPT-4o while using less than 1% of the training data, outperforming strong open-source baselines like Qwen2-VL.

### Strengths
1. The use of error-rate filtering and alternating SFT/GRPO training enables the model to automatically align task difficulty with model competence.
2. Demonstrates substantial performance gains and shows cross-task generalization on multiple benchmarks.

### Weaknesses
1. The first claim lacks sufficient examples or statistical justification and appears somewhat counterintuitive. Existing works on multimodal self-evolution typically revolve around data evolution and iteration based on image semantics, where visual and textual modalities are mutually complementary. However, the paper states that “Prior methods often address visual and textual components in isolation,” which is not very convincing and precise.
2. Although a model-based error filtering mechanism is introduced, there will still inevitably be residual noisy data. How does the framework further ensure the reliability and cleanliness of the iteratively generated data, _e.g._, incorrect mathematical formulas, variable confusion, or hallucinations, that appear during the iterative process?
3. The title emphasizes “self-improving reasoning on multimodal data,” but in practice, the work only addresses diagram-based mathematical reasoning tasks, with a relatively small dataset scale. Moreover, when scaled up, is the framework still efficient and cost-effective?

### Questions
Please see the Weaknesses.

### Soundness
3

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
This paper proposes C2-Evo, a closed-loop self-improvement framework for MLLMs that jointly evolves both training data and model capabilities, with a focus on diagram-based mathematical reasoning. It addresses two key problems: the mismatch between visual and textual complexity in multimodal datasets, and the misalignment between task difficulty and model ability during iterative training. 

The main contributions include: (i) a cross-modal data evolution loop generating progressively complex visual-text problems; (ii) a data-model evolution loop that adapts training difficulty to model performance with error-rate filtering; (iii) empirical experiments showing significant performance gains with minimal data; and (iv) plans to release code, models, and datasets. Multiple baseline comparisons are provided for open-source and closed-source reasoning models.

### Strengths
1. Introduces a joint co-evolution mechanism for both data and model, addressing an important gap in MLLM self-improvement research.  
2. Demonstrates performance improvements across multiple reasoning benchmarks while requiring <1% of baseline training data.  
3. Shows generalization beyond geometry, to mathematical function reasoning and broader multimodal benchmarks.

### Weaknesses
1. Evaluation primarily focuses on geometry problems, which may limit applicability claims to other complex multimodal domains.  
2. Heavy reliance on GPT-4o as an oracle for data generation may reduce accessibility for researchers without closed-source model access.  
3. Limited ablation on individual components of the framework (e.g., impact of each guiding principle in question generation) restricts the interpretability of improvements.

### Questions
1. How sensitive is C2-Evo’s performance to the choice of the oracle model, and can similar results be achieved with open-source or smaller models?  
2. Could you provide a detailed ablation separating the contributions of visual complexity evolution vs. textual complexity evolution?  
3. How does the computational cost of iterative evolution training compare to static dataset fine-tuning?
4. Missing reference: MMEvol: Empowering Multimodal Large Language Models with Evol-Instruct (ACL 2025)

### Soundness
3

### Presentation
3

### Contribution
2
