# Breaking Static Paradigms: A Mutual Evolution Framework for Edge-Cloud Model Collaboration

- Decision: Reject
- Scores: 2, 6, 2, 4, 8

## Abstract
To simultaneously achieve high performance and low latency, the paradigm of edge-cloud model collaboration, where Large Language Models (LLMs) are deployed on the powerful cloud and Small Language Models (SLMs) on the resource-limited edge devices, has garnered great attention recently. However, a key limitation of current edge-cloud architecture is its static nature, which hinders
the dynamic integration of new knowledge. More specifically, existing methods typically update the system by directly retraining the cloud-based LLM with edge-side newly collected data, which not only increases communication overhead but also neglects available computing power and data accessibility on edge devices. To tackle this challenge, we propose a novel mutual Evolution framework for edge-cloud model Collaboration called CoEvo that enables both cloud-side LLM and edge-side SLMs to update with new knowledge continuously. The cloud-based LLM can enhance edge-side SLMs through credible Chain-of-Thought (CoT) based knowledge distillation to improve its general understanding capabilities. Once the edge-side SLMs collect new domain-specific knowledge and optimize themselves locally, they will specifically enhance the cloud-based LLM via a credible probability matrix predicted on a few samples without uploading all raw data. Through this mutual evolution, the system can achieve continual optimization of the cloud and edge-side models and promote real-world deployments. Experimental results demonstrate a considerable performance gain of our edge-side SLMs against existing methods on the target dataset, with the cloud-side LLM also achieving a notable improvement over the base model.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper proposes CoEvo, a trainable framework for edge-cloud computing that finetunes both the cloud-based LLM and the edge-based LLM to improve performance. In particular, the LLM first distills knowledge in the SLM, weighting traces by their confidence of the LLM. Then, the SLM is first locally finetuned further and then used to update the LLM. For this purpose, the SLM generates probability matrices that are then sent and used by the LLM, only including matrices that have high probability generations. Results show that CoEvo outperforms baselines.

### Strengths
Unfortunately, I could not find any strengths of the work.

### Weaknesses
The paper suffers from several critical weaknesses and various weaknesses. I have split my concerns between these types and also included remarks, which are small points that have not influenced my judgment. I do note that due to many explanations missing (also a weakness), some of these points might come from misunderstanding on my part. The authors are free to refute them with a thorough explanation if they feel they are based on a misunderstanding.

**Critical Weaknesses**
- **The presented method seems far from applicable to any edge-based scenario**. The discrepancy between the proposed method and the proposed use-case are very high. Among others, here are a few indications that this method would never be practical:
   - If one can finetune an 8B model on edge-devices, one can also run a 70B model there. The compute required for finetuning are way higher than for inference, but the authors make no note of this, pretending it is easy for an edge-device to finetune an 8B model, which is most definitely not the case.
   - The evaluation does not actually do anything that is relevant to edge-based scenarios. The evaluation takes place on some standard benchmarks and an end-to-end evaluation is never performed. None of the benchmarks are somewhat applicable to any real-world scenario I can think of for edge-based devices (would be surprising if one user has GSM8k data, one has MMLU data, ...). 
   - The framework is limited to final-answer or multiple-choice based benchmarks. This limits its application significantly, especially because the best use case for edge-based computing is personalization, which cannot be done with multiple-choice answers.
   - There is nothing in the paper that explains how to relegate complex queries from the SLM to the LLM. This is a key missing aspect for the application the authors are arguing about. Without it, we essentially just have two separate models that do their own thing.
- **The method significantly increases communication overhead and does not improve user privacy, despite claims.** The problem regards the size of the probability matrix: for a typical vocab size of 30,000 and relatively short reasoning response of 100 tokens, the authors would send 3 **million** floats to the cloud for each samples on which they want to train (approx 10MB). Furthermore, the data sample can be simply obtained from the probability matrix by argmaxxing each column, it does not give any extra privacy guarantees.
- **Performance gains are minimal.** Performance improvements are not noteworthy and less than 2% almost everywhere.

**Weaknesses**
- **The paper uses outdated models, techniques, and datasets.** None of the used datasets are really used in practice anymore. The models used are also outdated. Furthermore, techniques such as CoT are outdated, this technique has now been finetuned specifically into models. Requiring models to answer in a single token without CoT is outdated and should never be used anymore, except (maybe) for simple classification tasks (which for instance GSM8k is not). 
- **Explanation about many critical aspects are missing.** There is an enormous amount of explanations missing that need to be filled out. For instance,
   - It is never explicitly mentioned what the training data is. I am currently assuming this is the training data associated with each benchmark? This would not be great, see later weakness.
   - It is not specified what training method is used for several components. For instance, the self-training of the SLM from $M_1$ to $M_2$ is not specified, other than that it is done. What is the data used here? On what traces is it finetuned? Does it do normal SFT? If so, how are the labels obtained (in most edge-based practical scenarios, labels are not available, since users only ask questions to LLMs that they don't know the answer to). Furthermore, it is unspecified what the loss function is that is used to train the LLM in the edge-to-cloud scenario: KL-divergence with respect to the probability matrix?
  - The setup for the paper is not properly identified: What are the assumption on the compute for the cloud and edge devices, what kinda tasks is the method applicable to? For instance, in section 3 the authors talk about labels without first having properly explained the method is only applicable to classification datasets.
  - Are the answers and reasoning in any way mixed in training the SLM in (3.2)? Since CoEvo only requiring a single inference step to classify things, what is the use of training on reasoning traces?
   - In 3.2, it is never mentioned how confidence is measured. Only in the appendix, it becomes clear from the prompt that this is self-reported confidence with some custom prompt. Furthermore, the prompt in this appendix asks to also provide the answer. How is this answer used? Does it need to match the answer provided in the single-step Q->A?
- **3.2 is not very novel, as it is a standard distillation technique.** The method distils the LLM into the SLM with some extra selection based on confidence. However, the plot in 2 (and the general literature) indicate that this is a rather poor selection method; confidence is only slightly higher for correct answer.
- **Training takes place on training sets (?) of benchmarks.** LLMs and SLMs should be used to handle a variety of questions. While an evaluation setup that contains a training set and test set with the exact same distribution is fine as a toy setup, it is not really applicable to any real use case.  
- **Extra inferences required for the scheme significantly increase compute requirements.** Edge-devices are compute constraints. Yet, the method seems to not take this into account. Not only is finetuning directly performed on edge-nodes, majority voting is also performed on each sample, requiring many more inferences compared to standard operation procedures. No analysis of this extra compute requirement is mentioned or discussed.

### Questions
See above

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the limitation of static knowledge in current edge-cloud collaborative language model architectures, where powerful LLMs reside in the cloud and smaller SLMs operate on edge devices. Existing methods for updating these systems often involve retraining the cloud LLM with raw data collected from the edge, incurring high communication costs, latency, privacy risks, and underutilizing edge compute capabilities. To overcome this, the authors propose CoEvo, a framework enabling mutual evolution where both the cloud LLM and edge SLMs continuously update with new knowledge. Experiments using Llama3 8B (edge) and 70B (cloud) on datasets like MMLU, CQA, GSM8K, and WinoGrande show that CoEvo improves the performance of the edge SLM compared to various baselines and also enhances the cloud LLM's performance compared to its base version, often achieving gains comparable to non-private SFT.

### Strengths
The core contribution is the introduction of a bidirectional, continuous learning framework for edge-cloud collaboration, moving beyond static deployments or simple cloud-centric updates. This mutual evolution concept is novel and addresses a significant limitation of prior work. The framework directly tackles major issues in edge-cloud systems: incorporating new knowledge dynamically, reducing communication overhead, enhancing privacy, utilizing edge compute resources for local updates, and mitigating catastrophic forgetting in the cloud LLM. Experiments demonstrate clear performance improvements on both the edge SLM (outperforming strong CoT and fine-tuning baselines) and the cloud LLM (achieving gains comparable to non-private SFT, while better preserving general capabilities). The ablation studies effectively validate the contribution of the key components (confidence weighting, sampling, filtering).

### Weaknesses
CoEvo involves multiple stages (cloud-to-edge, edge-to-cloud) and several sophisticated sub-components (confidence scoring, CoT generation, majority voting, probability matrix extraction, dual-threshold filtering). This introduces significant implementation complexity compared to simpler update strategies. The cloud-to-edge distillation relies heavily on the LLM's self-reported confidence being a reliable indicator of correctness/quality. While Figure 2 provides some justification, confidence calibration in LLMs is known to be imperfect and can vary across models and domains. Poor calibration could negatively impact the SLM distillation.

### Questions
How sensitive is the cloud-to-edge distillation performance to the calibration quality of the LLM's confidence scores? Have you experimented with different methods for obtaining or scaling these confidence scores?

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
3

### Summary
The paper proposes CoEvo, a bidirectional edge-cloud learning framework. Different from the traditional point of view, where the edge uploads data to the cloud, CoEvo considers a self-evolutional strategy on the edge. In the cloud-to-edge phase, an LLM distills to an on-device SLM using a confidence-weighted CoT loss. In the edge-to-cloud phase, the SLM uploads filtered probability matrices using dual thresholds on top-k mass and max probability. Experiments on four benchmarks show modest improvements over inference baselines and small gains over strong finetuning baselines.

### Strengths
- This work presents an intuitive idea that is easy to implement.
- Results show that CoEvo outperforms most inference baselines and is competitive with SFT.

### Weaknesses
- I don't see this approach to be much practical. 

First of all, we shouldn't expect edge devices to do fine-tuning by themselves. Edge devices are resource-constrained devices, and available compute power should be prioritized for inference. Even for finetuning, we can expect those updates to be done in a non-real-time manner (such as during maintenance or during less-service hours). I don't see a need to address the immediate latency issue. Second, the privacy of the user data is also not guaranteed. The probability matrices essentially reveal the user data, and I don't see how the data can be protected. Third, I'd say that matrices are much more communication-intensive than raw data.

- The idea of selective distillation is not novel.

In previous works [1], [2], this idea has been proposed.

[1] Chen et al. Towards Robust and Efficient Cloud-Edge Elastic Model Adaptation via Selective Entropy Distillation, ICLR 2024.

[2] Shao et al. Selective Knowledge Sharing for Privacy-Preserving Federated Distillation without A Good Teacher, Nature Communications.

### Questions
1. What is the exact payload size per edge-to-cloud update? How efficient is it to transmit a matrix compared to raw data?
2. Can you run membership inference or input reconstruction attacks against uploaded matrices to check if the claim over user privacy is held?
3. Please address the weakness. In particular, why is there a need for an immediate edge-side update, and why can't we let more powerful cloud services update edge models, while only transmitting user data from edge to cloud and model weights from cloud to edge?

### Soundness
1

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
4

### Summary
This work proposes CoEvo, a mutual evolution framework for edge-cloud model collaboration that enables continuous learning of new knowledge. It uses credible Chain-of-Thought distillation to transfer knowledge from cloud LLMs to edge SLMs, and credible probability matrices to feedback domain-specific knowledge from edge to cloud, achieving performance improvements on both sides while reducing communication overhead.

### Strengths
+ CoEvo introduces a mutual evolution framework for edge-cloud collaboration, breaking static paradigms by enabling both cloud LLMs and edge SLMs to continuously learn and improve from new domain-specific knowledge dynamically.
+ The framework incorporates confidence scores to weight and filter knowledge transfer in both directions, ensuring only high-quality information is distilled. This credibility mechanism mitigates the impact of erroneous outputs and improves training effectiveness.
+ CoEvo reduces data transmission overhead by performing local updates on edge devices and uploading only filtered probability matrices instead of raw data, achieving comparable performance to centralized training approaches.

### Weaknesses
-  Technical contribution is limited considering the existing work of mutual knowledge distillation between cloud and edge, such as COLLA proposed by Lu et al. in SEC 2019. The core techniques—confidence-weighted knowledge distillation and probability matrix filtering—are incremental extensions of existing mutual distillation methods. The framework primarily combines established techniques rather than introducing fundamentally novel mechanisms for edge-cloud collaboration.

- Fine-tuning billion-parameter models (Llama3-8B) on memory-constrained edge devices like smartphones is currently infeasible. The paper lacks discussion of computational requirements, memory constraints, and practical deployment challenges on real edge hardware.

- Experiments only evaluate one model pair (Llama3-8B/70B) across four datasets. The generalizability to other model architectures, size ratios, and diverse application domains remains unvalidated, limiting confidence in the framework's broad applicability.

- The paper lacks thorough analysis of critical design choices, such as convergence behavior across multiple iterations, robustness to noisy edge data, and failure modes. The filtering thresholds appear dataset-specific without principled selection guidelines.

### Questions
Please see Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper addresses the static nature of current edge-cloud model collaboration frameworks, where Large Language Models (LLMs) on the cloud and Small Language Models (SLMs) on the edge cannot be dynamically updated with new knowledge. The authors highlight that existing update methods, which involve uploading new edge-side data for retraining, suffer from high communication overhead, user latency, and data privacy concerns.

To solve this, the paper proposes CoEvo (Collaboration-Evolution), a novel mutual evolution framework. The framework consists of two main stages: Cloud-to-Edge: Credible CoT Knowledge Distillation & Edge-to-Cloud: Credible Probability Matrix Knowledge Distillation.

### Strengths
The paper introduces a novel two-stage "mutual evolution" framework that is well-designed and addresses a clear gap in existing edge-cloud architectures.
A major strength is the edge-to-cloud update mechanism. By optimizing the SLM locally and then uploading only filtered, "credible probability matrices" instead of raw data, the framework provides a practical way to learn from new domain-specific data while preserving user privacy.
The paper provides strong empirical support for its claims. It not only shows performance gains against strong baselines (Table 1) but also includes crucial ablation studies (Table 3) and a direct comparison against SFT (Table 4) to demonstrate its ability to mitigate catastrophic forgetting.

### Weaknesses
In the cloud-to-edge stage (Table 1), CoEvo underperforms the SPIN baseline on the GSM8K dataset. The paper hypothesizes this is due to the confidence metric struggling with tasks where multiple valid reasoning paths (CoTs) exist. This suggests the "credible CoT" mechanism may be less effective for domains with high solution entropy or creative reasoning tasks.
The edge-to-cloud stage requires the "resource-limited edge device" to perform several computationally non-trivial steps: local optimization (training M1 into M2), "multiple-sample voting" (Figure 3 suggests 10-20 samples), and then calculating and filtering probability matrices. The paper focuses on communication overhead savings but does not sufficiently analyze the computational overhead this process adds to the edge device.

### Questions
Could the authors elaborate on the computational overhead (e.g., latency, memory usage) of the edge-to-cloud pre-processing stage? Specifically, how does the cost of local optimization (M1 -> M2), multi-step sampling, and matrix filtering compare to a simple inference on the edge device? This seems like a significant workload for a "resource-limited" device.
Given the C2E stage's weaker performance on GSM8K and the hypothesis that this is due to multiple valid CoTs, does this limit CoEvo's applicability in more open-ended or creative tasks? Have the authors considered alternative methods for assessing rationale quality beyond a single confidence score for these domains?

### Soundness
3

### Presentation
4

### Contribution
4
