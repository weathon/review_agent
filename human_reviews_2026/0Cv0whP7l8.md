# Diagnosing and Mitigating Modality Interference in Multimodal Large Language Models

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 4, 6

## Abstract
Multimodal Large Language Models have demonstrated impressive capabilities across tasks, yet they often exhibit difficulty in distinguishing task-relevant from irrelevant signals—particularly in tasks like Visual Question Answering—which can lead to susceptibility to misleading or spurious inputs. We refer to this broader limitation as the Cross-Modality Competency Problem—the model’s inability to fairly evaluate all modalities. This vulnerability becomes more evident in modality-specific tasks—such as image classification or pure text question answering—where models are expected to rely solely on one modality. In such tasks, spurious information from irrelevant modalities often leads to significant performance degradation. We refer to this failure as Modality Interference, which serves as a concrete and measurable instance of the cross-modality competency problem, and we further design a perturbation-based causal diagnostic experiment to verify and quantify this problem. To mitigate modality interference, we propose a novel framework to finetune MLLMs, including perturbation-based data augmentations with both heuristic perturbations and adversarial perturbations, and a consistency regularization strategy applying on model outputs with original and perturbed inputs. Experiments on multiple benchmark datasets (image-heavy, text-heavy and multimodal tasks) and multiple model families with different scales demonstrate significant improvements in robustness and cross-modality competency, indicating our method's effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework to diagnose and mitigate modality interference in multimodal large language models (MLLMs)—a phenomenon where irrelevant or misleading modality signals degrade model performance. The authors define the broader cross-modality competency problem, identifying modality interference as a concrete instance. They propose (1) a perturbation-based causal evaluation that quantifies interference by injecting irrelevant signals into one modality, and (2) a fine-tuning strategy combining heuristic and adversarial perturbations with a consistency regularization objective. Experiments on image-heavy, text-heavy, and multimodal tasks (Mini-ImageNet, Caltech-101, OpenBookQA, MMLU, ScienceQA, MM-Bench, Seed-Bench) demonstrate significant robustness gains and improved unimodal reasoning without harming multimodal performance. The paper is technically solid and the framing is clear, though the conceptual advance is moderate.

### Strengths
1. The identification of modality interference as a measurable phenomenon and its connection to cross-modality competency is insightful.
2. The perturbation-based causal evaluation is well designed and empirically grounded, revealing meaningful vulnerability patterns.
3. The fine-tuning framework combining heuristic and adversarial perturbations with consistency regularization is practically effective.
4. The experiments are comprehensive across datasets, model sizes, and modalities, showing consistent improvements in robustness and generalization.

### Weaknesses
1. The theoretical depth is limited. The work largely integrates known ideas from causal probing and adversarial robustness without deeper theoretical analysis.
2. The causal effect metric ($\delta_{cp}$) is intuitive and a bit heuristic. It only captures prediction flips, not probabilistic changes or feature-level shifts.
3. The perturbation design may unintentionally change semantic content rather than purely isolate modality relevance.
4. The method introduces additional computation from adversarial training, and the efficiency trade-off is not discussed.
5. The paper emphasizes robustness metrics but provides little mechanistic insight into why the proposed perturbation strategy improves cross-modality alignment.

### Questions
1. How sensitive are the robustness gains to the perturbation strength \epsilon and the number of adversarial steps?
2. Can the proposed approach generalize to generative tasks or multimodal reasoning beyond multiple-choice formats?
3. How does the model behave under simultaneous perturbations in both modalities?
4. Could the causal effect be defined more continuously (e.g., using KL divergence on prediction distributions) to better quantify partial modality reliance?
5. Have the authors compared to other causal fine-tuning strategies such as counterfactual supervision or gradient orthogonalization?

### Soundness
3

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
4

### Summary
The paper proposes the cross-modality competency problem where a multimodal large language model (MLLM) fails to evaluate all modalities. A concrete example of this problem is modality interference, where MLLMs often use spurious information from irrelevant modalities when they are expected to rely solely on one-modality data. As a result, MLLMs often underperform on pure visual recognition and textual reasoning. The paper designs a perturbation-based experiment to verify and quantify this problem. Then, it proposes to fine-tune MLLMs with perturbation-based data augmentations. Experiments on image-heavy, text-heavy and multimodal tasks  and multiple model families verify the effectiveness of the proposed approach in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.

### Strengths
- The paper demonstrates that MLLMs are not robust under modality interference where different modalities are not aligned and only one modality is relevant to the task. This highlights an important robustness issue in MLLMs.

- The paper shows that using modality misaligned data for fine-tuning can mitigate modality interference and is effective in boosting both unimodal and multimodal reasoning abilities. 

- Experiments show that the proposed method is effective with different model families in different scales. 

- The paper is well-written and easy to follow.

### Weaknesses
- The technical contributions of the paper are limited. The proposed perturbation-based data augmentations are not novel, and it is expected to see performance improvements when incorporating data with modality interference. 

- Choice of the datasets: Why do the authors choose Mini-ImageNet and Caltech-101 as Image-heavy datasets and Open-BookQA and MMLU as text-heavy datasets? Would different choices of these datasets affect the performance of fine-tuned models on VQA tasks? 

- Concern on the generalizability of the proposed method: While it is expected to see performance improvement on image-heavy and text-heavy datasets when they are incorporated into the fine-tuning dataset, it is unclear how this would mitigate modality interference on unseen image-heavy and text-heavy datasets.

- The proposed adversarial perturbation to the latent embeddings is not effective as compared to the perturbation in the input space as shown in Table 2. Is this component essential to mitigating modality interference?

### Questions
- How to choose image-heavy and text-heavy datasets? Would different choices of these datasets affect the performance of fine-tuned models on VQA tasks? 
- Can the proposed method geeralize to unseen image-heavy and text-heavy datasets?
- Lines 264-265: How to choose $N_{img}$, $N_{text}$, and $N_{vqa}$ in practice to ensure effective modality interference mitigation?  How do different choices of these values affect the performance?

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
The paper first defines the Cross-Modality Competency Problem where existing Multimodal Large Language Models (MLLMs) are susceptible to misleading inputs especially in  modality-specific tasks, such as image classification or pure text question answering, where models are expected to rely solely on one modality. The paper, first benchmarks this across a range of models using a perturbation-based causal diagnostic setup. Perturbations include - for image-heavy tasks, unrelated scientific facts and misleading descriptions and - for text-heavy tasks are including a real, irrelevant image or a full black/white canvas image. Next, to improve upon this shortcoming, a novel framework to finetune MLLMs is presented which includes adversarial losses and a consistency regularizer strategy at the input and representation level.  Experiments on multiple datasets and models demonstrate significant improvements in robustness and cross-modality competency, indicating the method’s effectiveness in boosting unimodal reasoning ability while enhancing performance on multimodal tasks.

### Strengths
1. The definition of Modality Interference is well defined and the findings that model performance goes down due to sub-optimal integration information across modalities is interesting.
2. The motivation behind the proposed losses are well defined. 
3. The paper is well written and easy to follow.

### Weaknesses
1. The proposed losses are not very effective : As shown in the ablations in Table 2, FFT with VQA/AUG performs better than proposed losses. Examples being : LLaVA-1.5-13B - FFT with $D^{AUG}$ - on ScienceQA-IMG | Qwen2.5-VL-3B - FFT with $D^{VQA}$ - on  MM-Bench-EN.
2. Consistency of results : In Table 1, the drop in performance of models on Caltech 101 is quite high, for example LLaVA-1.5-7B, goes from (97.0 --> 57.4), but in Table 4 : the drop is much less on OCR images  (97.0 --> 92.8) ; this raises questions around. i) if modality interference is really a concern and/or ii) if the generalization results in Table 4 are correct.
3. Provided results are on 3B/7B/13B models; results on newer family of thinking models would help solidify the claims made about failure modes in the paper.
4. To truly evaluate the effectiveness of the proposed losses, evaluation against other adversarial attacks should also be presented.

### Questions
1. It seems from Table 1 that drop in performance on models is much larger on vision-heavy tasks (such as Mini-ImageNet) is much larger than language-heavy tasks (such as OpenBookQA) - why might this be the case?
2. What is the reasoning behind choosing these perturbations : 
i) unrelated scientific facts or misleading descriptions - for image-heavy tasks. 
ii)  semantically meaningful real images/ full black canvas/ full white canvas - for language-heavy tasks. 
3. Does the current evaluation of Modality Interference require Causal Modeling?
4. An ablation with/without the modality-specific binary mask for the adversarial loss will be interesting to see.

### Soundness
2

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
The paper investigates modality interference in MLLMs, particularly in tasks like Visual Question Answering. It defines modality interference as the negative impact of irrelevant modalities on unimodal tasks and quantifies this issue through perturbation-based causal diagnostics.

To mitigate this interference, the authors introduce a new fine-tuning framework that incorporates data augmentation and consistency regularization strategies to improve model stability across different inputs. Experimental results demonstrate significant enhancements in robustness and overall performance.

### Strengths
Innovative Concept: The paper introduces the notion of the Cross-Modality Competency Problem, providing a fresh perspective on modality interference in multimodal large language models. This innovative approach contributes new insights to the field.

Systematic Analysis: By designing a perturbation-based causal diagnostic experiment, the authors quantify the impact of modality interference, providing empirical evidence that enhances the scientific rigor and validity of the research.

Effective Solution: The proposed fine-tuning framework combines data augmentation with consistency regularization strategies, offering a practical solution to mitigate modality interference. This approach has been validated through significant improvements in robustness and performance across multiple benchmark datasets.

### Weaknesses
Potential Overfitting Risks: The use of perturbation-based data augmentation may introduce noise into the training process. While it aims to enhance robustness, there is a risk that the model might overfit to these perturbed examples, resulting in poorer generalization on clean, real-world data.


Lack of Comparative Baselines: The paper does not provide a comprehensive comparison against a wider variety of existing methods or models that address modality interference. Without robust baseline comparisons, it is difficult to ascertain the relative effectiveness of the proposed framework.


Limited Experimental Diversity: The experiments primarily focus on a small set of benchmark datasets, which may not capture the full range of conditions under which modality interference could occur. This limited range could restrict the generalizability of the findings to other real-world scenarios.

### Questions
How does the performance of the model vary with different configurations of the augmentations or regularization strategies?

### Soundness
3

### Presentation
3

### Contribution
4
