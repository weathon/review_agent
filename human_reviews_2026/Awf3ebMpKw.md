# Expert Merging: Model Merging with Unsupervised Expert Alignment and Importance-Guided Layer Chunking

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 4

## Abstract
Model merging, which combines multiple domain-specialized experts into a single model, offers a practical path to endow Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) with broad capabilities without the cost of joint training or serving many models. However, training-free methods rely on hand-tuned coefficients, whereas training-based methods primarily align parameters rather than downstream task behavior and typically treat all layers uniformly, ignoring inter-layer heterogeneity. We introduce Expert Merging, a training-light method that learns a small set of layer-wise coefficients using only unlabeled calibration data. The coefficients are optimized to explicitly align the merged model’s hidden states and logits with those of the corresponding experts, with a coefficient regularizer for stability and task-weighted losses for controllable trade-offs. To capture inter-layer variation, Expert Merging++ augments this design with importance-guided chunking: a normalized layer-importance metric, derived from learned coefficients, task-vector magnitudes, and parameter counts, allocates more chunk-wise coefficients to high-importance layers while keeping low-importance layers lightweight. The result is a label-free, parameter-efficient, and scalable approach to multi-expert model merging across LLMs and MLLMs. Across MLLM backbones (InternVL and Qwen2-VL) and the LLM backbone (Mistral), our method surpasses strong training-free and training-based merging baselines, with Expert Merging++ delivering further gains and, in some cases, even exceeding supervised Mixture Training. Our code is available at https://github.com/Littleor/ExpertMerging.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This manuscript addresses the limitations of existing model merging methods—where training-free methods rely on manual coefficients and training-based methods overlook task behavior alignment and layer heterogeneity—and proposes two lightweight training-based model merging schemes: Expert Merging and Expert Merging++.
The former learns layer-wise coefficients using only 5–10 unlabeled calibration samples. It explicitly aligns the hidden states and logits of the merged model with those of the expert models, and introduces coefficient regularization to ensure optimization stability, as well as a task weight loss to achieve controllable cross-domain trade-offs.
Building on the former, the latter combines learned coefficients + task vector magnitude + parameter count to construct a normalized layer importance metric. It allocates more block-wise coefficients to high-importance layers and keeps low-importance layers lightweight, further adapting to layer heterogeneity.
The manuscript validates these two methods across multi-task scenarios on LLM and MLLMs. Both methods outperform training-free and existing training-based baselines, with Expert Merging++ delivering better performance—it even surpasses supervised Mixture Training in some scenarios.

### Strengths
1 The authors' experiments cover 3 different models (1 LLM and 2 MLLMs) as well as multiple datasets, making the experimental design relatively comprehensive.

2 The proposed method is relatively lightweight and relies on only 5–10 samples.

### Weaknesses
1 Regarding the selection of unlabeled calibration samples, the authors do not provide clear details. Are the samples selected randomly? If random selection is adopted, the randomness of sample selection may reduce the reproducibility of the results.

2 For the selection of hyperparameters α and κ, although the authors provide the final values in the appendix, there is a lack of ablation experiments for hyperparameters and the range of parameter selection. This makes it impossible to further verify the parameter sensitivity.

3 As shown in Figure 2, the authors also merged the base model. What would the result be if only the expert models were merged?

4 What are the specific detailed settings for the optimization process of the objective function? For example, what are the values of parameters such as the number of iterations?

5 It is suggested that the authors classify the different comparative methods in Tables 1, 2, and 3 by type, such as training-based and training-free types

### Questions
see weakness

### Soundness
3

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
This paper proposes a new method, Expert Merging (and its extension Expert Merging++), for merging multiple domain-specialized expert models into a single large language model (LLM) or multimodal large language model (MLLM). The approach is “training-light,” requiring only a small set of unlabeled calibration data to learn layer-wise (and chunk-wise) coefficients that align the merged model’s hidden states and logits with those of the experts. The method introduces a layer-importance metric to guide parameter allocation, aiming to improve performance and efficiency. Experiments on several LLM and MLLM backbones (Mistral, InternVL, Qwen2-VL) show that the proposed method outperforms strong training-free and training-based baselines, and in some cases even surpasses supervised mixture training.

### Strengths
* The paper addresses a real and growing problem—how to efficiently combine multiple expert models without the cost of joint training or serving many models.
* The method requires only a handful of unlabeled samples per task, making it highly practical for scenarios where labeled data is scarce or unavailable.
* By aligning both hidden states and logits, the method goes beyond parameter-space alignment and attempts to preserve downstream task behavior.

### Weaknesses
* The method assumes that expert models are complementary, but does not analyze or address what happens when experts have conflicting or overlapping capabilities. There is no discussion of negative transfer or catastrophic forgetting.
* The method relies on aligning hidden states and logits, but there is little theoretical analysis or justification for why this should guarantee optimal downstream performance, especially when expert models may have conflicting behaviors.

### Questions
* Please provide more empirical or theoretical evidence that your layer-importance metric is robust and generalizes across architectures. For example, does it correlate with actual task performance sensitivity per layer?
* Can you provide more theoretical justification for the choice of alignment objectives (hidden states and logits)? Under what conditions does this guarantee downstream task preservation?
* Please analyze and report what happens when expert models have overlapping or conflicting capabilities. Does the merged model degrade on some tasks? Are there cases of negative transfer?
* In your experiments, the merged model (Expert Merging/Expert Merging++) sometimes outperforms the individual domain expert models on certain tasks. Could you provide a detailed explanation or analysis for why the merged model is able to surpass the best domain expert (or mixture training), given that it is only aligning to the experts' hidden states and logits using a small amount of unlabeled data? Is this effect consistent across different tasks and domains, or could it be due to overfitting, ensembling effects, or other factors? Please clarify the underlying mechanism and provide supporting evidence or ablation studies.
* Moreover, is Expert Merging (for example, on InternVL2.5 grounding tasks) effectively a performance ceiling, or—given the same data budget (i.e., access to all domain data)—could a domain-targeted method or supervised fine-tuning achieve strictly better results in that domain? Please clarify the underlying mechanism and provide supporting evidence or ablation studies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper addresses the issues of task alignment and inter-layer heterogeneity in model merging, and presents the training-light Expert Merging and Expert Merging++ methods. Expert Merging learns layer-wise coefficients to align the merged model’s hidden states and logits with those of expert models using unlabeled data, while Expert Merging++ introduces importance-guided, chunk-wise coefficients to further capture inter-layer heterogeneity. Extensive experiments on both LLMs (Mistral with Chat, Math, and Code experts) and MLLMs (InternVL2.5 and Qwen2-VL across five visual reasoning tasks) demonstrate that the proposed methods consistently outperform training-free and prior training-based baselines, including WUDI v2, AdaMerging++, and even supervised Mixture Training in some cases.

### Strengths
- **Well-motivated issues and clear solutions:** The paper identifies two critical limitations in model merging research, i.e., task alignment and inter-layer heterogeneity, and proposes direct and effective solutions for them.

- **Comprehensive evaluation and insightful analysis:** The experiment is well-designed and comprehensive. Its evaluation spans multiple architectures, diverse tasks, and includes comparisons against many strong baselines. The consistency of improvements strengthens the initial claims. Moreover, this paper presents some meaningful discussions, such as the layer-wise importance analysis.

### Weaknesses
- **The proposed methods are somewhat incremental:** While Expert Merging and Expert Merging++ are both effective, hidden-state and logit alignment losses are well established in knowledge distillation and model compression. If the novelty and effectiveness of the proposed methods are attributed mainly to the combination of techniques such as layer-wise coefficient learning and importance-guided chunking, the contribution seems somewhat incremental.

- The proposed methods involve some empirical tricks/techniques, such as the selection of hidden alignment layer and task weights, would this affect their adaptation and transferability?

- In addition, the proposed methods involve a considerable number of coefficients/hyperparameters that require tuning, would this affect their robustness?

### Questions
1. Why does aligning hidden states + logits offer better task alignment than entropy-based or interference-based objectives? Beyond the intuitive justification, is there any strong analysis or evidence?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper addresses the problem of model merging, assuming a pool of K pre-trained models is available, and these have *expertise* on certain tasks. From the perspective of both training-free methods and optimization-based ones, problems related to the inter-layer heterogeneity and variation of the neural nets are not yet solved. For this reason, a new framework consisting of two methodologies (Expert Merging & Expert Merging ++) is proposed, whose main feature is to describe a new procedure of several steps, where the key ideas are i) learn per-layer coefficients within alignment of representations and ii) estimation of the layer importance. Experimental results with LLMs (i.e., Mistral-7B-v0.1 and InternVL2.5-1B-Instruct) show that the performance is on-pair and sometimes better than a wide range of SOTA methods for model merging.

### Strengths
- **SOTA & Related Work:** I do think the paper and the proposed method are correctly based on evidence, driving the concerns around the limitations of current SOTA methods. I particularly enjoyed reading the division of such methods into the two families, one for the training-based and the non-training-based methods. That was clear enough to me and for experienced readers interested in these topics. In general, the introduction and presentation of the model merging problem are clear and understandable for the ICLR audience.
- **Critical goals and key points:** I also perceived a very powerful identification of the main goals to solve for the model merging challenge. For instance, alignment of logits, layer importance, or just comprehensive results on some multi-task validation test. The mission is certainly not easy, but sometimes you don't see these sorts of goals and issues as clearly stated as they are in this paper on other model merging contributions.
- **Alignment of losses:** One of the parts that I was more positively surprised about and which I would've liked to know more about is the subsection on alignment of losses and logits. Although I didn't clearly understand the motivations for making the choices on Eq. 1 and Eq. 2, I see the power of the proposed approach, and this is something I've seen repeatedly missing in some minor works in the last years on model merging.
- **Empirical results:** The paper has a significant empirical contribution, and experiments have the will to rigorously demonstrate that the performance of the ExpertMerging/ExpertMerging++ is worth being used and superior to other SOTAs. I am, however, not entirely convinced of this superiority, looking at Tables 1 and 2, or at least I get a bit lost on the heterogeneity of results and some lack of clear analysis and identification of issues/strengths related to the performance

### Weaknesses
- **Arithmetic merging/linear regression between models:** My main concern comes from the fact that the proposed model basically merges parameters according to a linear regressor with real-valued coefficients, as shown in Eq. 5 and in section 3.2, second paragraph. In that sense, I don't see much novelty or at least i am not convinced of the great progress that these propositions make, apart from being applied to big LLMs or MLLMs (i.e. Mistral one) from an academic point of view.
- **Specification and highlighting issues in SOTA models:** Unfortunately, I don't see great research contribution in the way the work and procedures are introduced. I do miss understanding (at an Equation level) what the problems and limitations of, for instance, AdaMerging or similar baselines. In the manner that section 3 describes things and introduces procedures, one can barely understand the motivations and issues that drove each one of the modelling/learning decisions taken. From this lens, it is just more of a description of engineering features and steps taken during the development of some tool with a reasonable purpose.
- **ExpertMerging/ExpertMerging++:** I had a hard time understanding if they can be combined, one is the enhanced version of the other, or just focus on different challenges. Sorry about this, if I may have missed something, but it's not clear to me, and I would need additional clarification.
- **Performance and results:** I don't have doubts about the honest empirical results provided apart from the fact that the code and software are not accessible to the reviewer ( I mentioned this in the next section). The main problem that I see is that I don't perceive superior performance from the experiments provided for a method and procedure, which I don't know how to reproduce or to understand from a research decision-making viewpoint.

### Questions
I certainly do not agree with the statement "*We will release the source code upon acceptance*". Understandably, under industrial competition and similar issues, this reviewer can understand certain practices or not releasing full systems to *operate* models at scale; however, I do think not providing any source code, and promises of doing it conditioned on acceptance, is, in general, a bad practice. I prefer not to raise additional concerns or points regarding this issue with the submission, but I do recommend that authors review the code of ethics of ICLR, particularly the sections on *standards of scientific excellence* (related to reproducibility) and on *trustworthy and transparent research.

### Soundness
2

### Presentation
3

### Contribution
2
