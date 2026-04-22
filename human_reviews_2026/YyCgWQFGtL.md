# Athena: Enhancing Multimodal Reasoning with Data-efficient Process Reward Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 8, 4, 4

## Abstract
We present Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward score for each step in solving complex reasoning problems. Developing high-performance PRMs typically demands significant time and financial investment, primarily due to the necessity for step-level annotations of reasoning steps. Conventional automated labeling methods, such as Monte Carlo estimation, often produce noisy labels and incur substantial computational costs. To efficiently generate high-quality process-labeled data, we propose leveraging prediction consistency between weak and strong completers as a criterion for identifying reliable process labels. Remarkably, Athena-PRM demonstrates outstanding effectiveness across various scenarios and benchmarks with just 5,000 samples. Furthermore, we also develop two effective strategies to improve the performance of PRMs: ORM initialization and up-sampling for negative data. We validate our approach in three specific scenarios: verification for test time scaling, direct evaluation of reasoning step correctness, and reward ranked fine-tuning. Our Athena-PRM consistently achieves superior performance across multiple benchmarks and scenarios. Notably, when using Qwen2.5-VL-7B as the policy model, Athena-PRM enhances performance by 10.2 points on WeMath and 7.1 points on MathVista for test time scaling. Furthermore, Athena-PRM sets the state-of-the-art (SoTA) results in VisualProcessBench and outperforms the previous SoTA by 3.9 F1-score, showcasing its robust capability to accurately assess the correctness of the reasoning step. Additionally, utilizing Athena-PRM as the reward model, we develop Athena-7B with reward ranked fine-tuning and outperforms baseline with a significant margin on five benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Athena-PRM, a multimodal process reward model (PRM) designed to evaluate the reward scores of reasoning steps within complex tasks. The authors assert that annotation costs can be reduced by generating pseudo step-level labels through the predictive consistency between weak and strong completers. To further enhance PRM performance, the study proposes ORM initialization and negative data upsampling. Experimental results across multiple reasoning benchmarks reportedly surpass existing models, with the proposed method achieving state-of-the-art outcomes on both the VisualProcessBench and math-related datasets.

### Strengths
This paper addresses the cost and scalability issues of step-level reward labeling by weak-strong completers, a key challenge in process supervision. This work also demonstrates significant improvements across several benchmark datasets.

### Weaknesses
- The authors repeatedly emphasize the results obtained using 5,000 samples but fail to specify the complexity of each sample or the cost of generating completions from the strong and weak models. This makes the argument for efficiency unconvincing.
- The data appears to be cherry-picked from the 60k sample pool, as there is minimal performance gain when scaling from 5k to 60k samples. An analysis of the performance growth curve is needed to validate the scaling properties.
- There is a discrepancy between the model used for validation within the strong-weak model framework and the actual target model used in the experiments. Therefore, it remains questionable whether the method is genuinely effective for the target model.
- The paper lacks a comparative analysis with other data synthesis strategies and does not include ablation studies on relevant components. Consequently, it is unclear how much the core data sampling strategy actually contributes to the observed performance.
- The experiments are primarily focused on mathematical reasoning datasets (e.g., WeMath, MathVista). It is worth investigating whether the method can still generate high-quality Best-of-N (BoN) samples for reasoning tasks in other domains, such as commonsense and science (e.g., by testing on benchmarks like M3CoT or EMMA).
- The study appears to exclusively use Qwen2.5-VL-7B as the training backbone, which makes it difficult to assess the general applicability of the data. This approach risks biasing the evaluation towards the capabilities of a specific large model rather than the intrinsic quality of the PRM data itself.

> My inquiry involves multiple questions regarding an extensive set of experiments. I will adjust the score to reflect the degree to which your response resolves these points.

### Questions
- Could the authors quantify the complexity of each sample and provide the computational cost associated with generating completions from both the strong and weak models? Without such data, the argument for the method's efficiency is difficult to substantiate.
- Could the authors present a performance growth curve for scaling from 5,000 to 60,000 samples? The minimal performance gain observed suggests a potential selection bias in the data, and an analysis of scaling properties would be necessary to validate the method's scalability.
- A discrepancy is noted between the model used for validation within the strong-weak framework and the target model employed in the final experiments. Could the authors justify this choice and provide evidence that the method's effectiveness indeed transfers to the final target model?
- To what extent does the core data sampling strategy contribute to the observed performance gains? A comparative analysis against other data synthesis methods, would be crucial to isolate the contribution of the proposed technique.
- How effective is it to use data distilled solely from a weak or a strong completer?
- Given the experimental focus on mathematical reasoning, it is unclear whether the method can generate high-quality Best-of-N (BoN) samples for reasoning tasks in other domains. Have the authors considered evaluating its efficacy on commonsense or scientific reasoning benchmarks, such as M3CoT or EMMA?
- The study's reliance on Qwen2.5-VL-7B as the sole training backbone makes it difficult to assess the general applicability of the data. Have the authors considered experiments with other model architectures to disentangle the intrinsic quality of the PRM data from the capabilities of this specific model?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes Athena-PRM, a multimodal process reward model (PRM) designed to be data-efficient. The core problem it addresses is the prohibitive cost of acquiring step-level annotations for training PRMs. The authors introduce a data-filtering method that leverages prediction consistency between a weak completer and a strong completer to identify high-quality, reliable process labels from automated methods. Using this method, they demonstrate that a PRM trained on only 5,000 high-quality samples can achieve performance comparable to or better than models trained on 300K noisy samples. The paper also introduces two effective training strategies: ORM initialization which uses a pretrained outcome reward model as a starting point and negative data up-sampling to handle label imbalance. The resulting Athena-PRM is validated across three scenarios—test-time scaling (Best-of-N), direct step evaluation, and reward-ranked fine-tuning, where it significantly boosts the performance of various policy models and achieves state-of-the-art results on the VisualProcessBench benchmark.

### Strengths
The "consistency filtering" technique using weak and strong completers is an intuitive and clever way to distill high-quality labels from noisy, automated sources. The finding that a mere 5K high-quality samples can rival a 300K-sample dataset is a significant contribution, addressing the bottleneck of annotation cost and computational expense in reward modeling. 

The authors conduct thorough empirical validation across different application scenarios, with Athena-PRM achieving state-of-the-art performance on VisualProcessBench and even surpassing proprietary models like GPT-4o, underscoring its versatility and effectiveness.

The paper is well-structured, and its claims are well-supported by ablation studies. Table 5 clearly disentangles the contributions of the key components: the high-quality Athena-5K data, the ORM initialization, and the negative up-sampling. This analysis confirms that each proposed strategy provides a tangible benefit, making the methodology easy to follow and justifying the design choices.

### Weaknesses
While the PRM is evaluated on policy models from different families (Qwen, InternVL, Ministral) , the PRM itself is a fine-tuned Qwen2.5-VL-7B. Furthermore, the data generation process is dominated by the Qwen family. This raises a minor concern about generalizability. It would be more convincing if the authors showed that the Athena-5K dataset could be used to train another effective PRM based on a different model family.

### Questions
The ablation in Table 5 is a key result, showing Athena-5K outperforms Random-5K. How does the performance of your consistency-filtered dataset (Athena-K) scale as K increases? For instance, does an "Athena-50K" dataset significantly outperform the MC-300K baseline, or do the returns on data quality diminish quickly after the 5K-sample mark?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes Athena-PRM, a data-efficient multimodal Process Reward Model (PRM) that scores each step in a reasoning chain. The model is leveraged in three scenarios: (1) test-time Best-of-N expansion, (2) direct step-correctness evaluation (VisualProcessBench), and (3) Reward-Ranked Fine-Tuning (RAFT) to produce Athena-7B.

### Strengths
The approach is well-motivated and demonstrates performance improvements.

### Weaknesses
While the proposed filtering strategy is the core contribution, it raises two significant concerns: Efficiency: The strategy is alarmingly inefficient, reducing 300K samples to a mere 5K. Bias: This aggressive filtering may introduce selection bias, potentially retaining only the most high-confidence correct and incorrect examples.

### Questions
The comparison in Table 3 is confounded. Given that the Qwen base model itself already provides a 3.0-point advantage over InternVL, the reported 3.9-point improvement for Athena-PRM cannot be cleanly attributed to the proposed method.

The baseline in Table 4 is inadequate. To properly isolate the PRM's contribution, the baseline must also be trained on the same filtered dataset, but with a randomly selected trajectory for SFT, rather than the PRM-selected one. Without this direct ablation, it is impossible to prove that the performance gain stems from the PRM's selection capability and not merely from the introduction of the filtered data itself.

Part of the paper are confusing. For example, Athena-300K  in line893 should be MC-300K?   Athena-5K should be a subset of MC-300K?+9.9 in line354 should be 10.9?

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
3

### Summary
This work proposes Athena-PRM, a multimodal process reward model that assigns step-level rewards to reasoning trajectories. To generate high-quality process labels efficiently, it introduces a consistency filter that keeps only steps whose Monte Carlo hard labels agree between a weak and a strong model. Two training strategies are explored: initializing PRMs from an outcome reward model, and up-sampling negative step labels. Athena-PRM is evaluated in three scenarios: Best-of-N test-time scaling, direct step judgment, and reward-ranked fine-tuning. Across seven multimodal and two text-only benchmarks, Athena-PRM shows consistent gains.

### Strengths
- The paper presents a simple and effective label-quality filter via weak/strong completer consistency. 
- This work achieves  strong multi-benchmark empirical performance. Three scenarios—BoN verification, direct step judgment, and RAFT—are specified and evaluated, demonstrating versatility.

### Weaknesses
- Limited validation scale and sensitivity of the consistency filter. The label accuracy validation uses only 50 queries , which is small and may not generalize; larger validation is not shown. Sensitivity to the number of MC samples T is not analyzed; all examples use T=8, yet hard labels depend heavily on T, affecting reliability.
- Misaligned RAFT baselines in “Evaluation on the Fine-Tuned Model” . The paper’s objective is to demonstrate Athena-PRM’s effectiveness, so RAFT should be compared against RAFT driven by alternative PRMs/ORMs (e.g., VisualPRM, text verifiers, ORMs), rather than against fine-tuned models with different training data and recipes.  The current setup conflates improvements from data selection, training pipeline, and model size with the reward model’s contribution, hindering clean attribution. 
- Mathematical and notation clarity issues: 1) Step segmentation/labeling details (“\n\n”, “+/-” labels) lack precise token-level application description, which is important for reproducibility. 2)Minor typos, e.g., “Tabele 6” in Sec. 2.2; duplicated “ϕ2w” in Table 6 caption.
-  Reproducibility. No evidence of code, data, or model release to support reproducibility and result authenticity.

### Questions
Regarding Eq. (1) and Eq. (2), should there be a negative sign (i.e., standard negative log-likelihood)? What is r’s range and activation (e.g., sigmoid to ensure r∈(0,1)) to avoid log-domain issues for log r and log(1−r)?

### Soundness
3

### Presentation
3

### Contribution
2
