# A Guardrail for Safety Preservation: When Safety-Sensitive Subspace Meets Harmful-Resistant Null-Space

- Avg Score: 6.40
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8, 4

## Abstract
Large language models (LLMs) have achieved remarkable success in diverse tasks, yet their safety alignment remains fragile during adaptation. 
Even when fine-tuning on benign data or with low-rank adaptation, pre-trained safety behaviors are easily degraded, leading to harmful responses in the fine-tuned models. 
To address this challenge, we propose GuardSpace, a guardrail framework for preserving safety alignment throughout fine-tuning, composed of two key components: a safety-sensitive subspace and a harmful-resistant null space. 
First, we explicitly decompose pre-trained weights into safety-relevant and safety-irrelevant components using covariance-preconditioned singular value decomposition, and initialize low-rank adapters from the safety-irrelevant ones, while freezing safety-relevant components to preserve their associated safety mechanism. 
Second, we construct a null space projector that restricts adapter updates from altering safe outputs on harmful prompts, thereby maintaining the original refusal behavior. 
Experiments with various pre-trained models on multiple downstream tasks demonstrate that GuardSpace achieves superior performance over existing methods.
Notably, for Llama-2-7B-Chat fine-tuned on GSM8K, GuardSpace outperforms the state-of-the-art method AsFT, reducing the average harmful score from 14.4\% to 3.6\%, while improving the accuracy from from 26.0\% to 28.0\%.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
When fine-tuning on benign data or with low-rank adaptation, pre-trained safety behaviors are easily degraded, leading to harmful responses in the fine-tuned models.
The paper introduces GuardSpace, a fine-tuning guardrail that preserves aligned LLM’s refusal behavior.
It comprises two key components:
1. It initializes the adapters based on the decomposed pre-trained weights in a safety-sensitive subspace.
2. Project learnable adapters onto the harmful-resistant null space so that the weight update will be nullified on harmful inputs.

### Strengths
- The design is clean, easy to plug into LoRA.
- It produces strong performance on jarmfulness and utility across models.

### Weaknesses
- Both the subspace and projector come from C computed on a sampled harmful set; if that set misses attack families, safety may not be preserved for those unseen modes.
- The paper only tests on SST-2, AGNEWS, and GSM8K, whereas safety issues usually show up in generation. So the current setup might not reflect the method that keeps utility when generating text. Also, since the authors follow the Yang et al. (2025) evaluation, which includes AlpacaEval, perhaps show us the performance on AlpacaEval.

### Questions
- Since both the subspace and projector come from a covariance built on a sampled “harmful” set, how robust is GuardSpace to unseen attack families?
- Most reported tasks are classification (SST-2, AGNEWS) and reasoning (GSM8K), but safety evaluation often shows up in regular generation. Do results hold on open-ended generation benchmarks (e.g., AlpacaEval)?
- What is the compute/memory cost to build the subspace and projector?

### Soundness
2

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
The paper proposes GuardSpace, a framework that preserves safety alignment throughout adaptation. It does so by first separating pretrained weights into safety-relevant and irrelevant components via covariance-preconditioned singular value decomposition. The safety-relevant weights are kept frozen while the safety irrelevant parameters are used to get the LoRA adapters for training. This helps prevent loss of safety-alignment that occurs during the later training stages. A null-space projector is proposed which constrains the optimization on safety outputs such that the adapters during LoRA training do not replace them with harmful ones. Experiments are conducted against state-of-the-art methods using a variety of models across diverse tasks, and results show significant reduction in harmfulness without much impact on overall task performance.

### Strengths
1. It tackles a relevant and timely problem.
2. The solution proposed is very good and innovative. Using basic matrix decomposition principles, the authors are able to tackle a very challenging problem. It may be extended for other use-cases apart from just safety-alignment as well.
3. The results are very good, and clearly outperform current state-of-the-art.
4. The experimental section has good ablations and analysis that are informative.

### Weaknesses
1. The solution reduces the number of parameters tuned during its optimization phase which in turn might make it more susceptible to over-fitting during fine-tuning - a significant problem for LLMs which are expected to generalize well.
2. The authors make leaps in explaining their derivation of the the null-space making the logic of why a certain step is required in some parts hard to follow.
3. The solution adds quite a bit of compute on top of LoRA.

### Questions
1. It would be great for the authors to add some intuitions and explanations on how they authors came up with this interesting approach.
2. How does this impact the performance of benchmarks that are not used in fine-tuning? For example - Llama-2-7B-Chat (base model before fine-tuning) has been evaluated on Toxigen and MMLU. After training using GuardSpace for SST2, does the Toxigen/MMLU results change? This will answer questions on generalizability.
3. How much compute overhead does this add? While this method is still cheaper than other types of non-PEFT training methods, how does it compare to LoRA exactly? Is it significant enough to gate-keep already expensive-to-train LLMs?

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes GuardSpace, a two-part guardrail for safe fine-tuning with LoRA: (1) a covariance-preconditioned SVD that probes a pretrained model with safety-triggering prompts to estimate a safety-sensitive subspace per linear layer, then freezes the safety-relevant directions and initializes LoRA adapters from the safety-irrelevant directions. Furthermore, the proposed method introduces a fixed null-space projector, built from the same harmful-prompt activations, that constrains adapter updates so they induce zero activation along harmful inputs, thereby preserving refusals throughout training. Conceptually, the preconditioned factorization highlights safety-relevant structure (large singular values) while allocating trainable capacity to low-sensitivity components. For experiments, the method yields strong safety–utility trade-offs across models and tasks. The ablations indicate the null-space projector is the primary driver of safety, while subspace-aware initialization offers additional gains with minimal utility cost.

### Strengths
1. The paper introduces a clear two-part mechanism: safety-sensitive initialization plus a harmful-resistant projector, which is principled and easy to compose with standard LoRA training.
2. Results on three datasets and multiple model families demonstrate consistent safety reductions without sacrificing utility.
3. Robustness analyses vary the fraction of poisoned data and the corpus/size used to build the projector, showing stability with only a few hundred prompts.
4. The technique is robust to different backbone families, suggesting the safety subspace captures transferable structure.
5. Implementation is straightforward and compatible with common PEFT libraries, lowering the barrier for adoption in practice.

### Weaknesses
1. The construction of the covariance matrix and projector relies on the choice and coverage of safety-triggering prompts, which may not span evolving jailbreak schemes.
2. The method focuses on linear layers with LoRA-style adapters, so it is unclear how well it extends to other PEFT forms or full-parameter fine-tuning.
3. There is limited cross-bench triangulation with attack-success metrics beyond a brief appendix reference.
4. The approach assumes harmful inputs share a stable null space with the sampled prompts, which may not hold under strong distribution shift or adaptive adversaries.

### Questions
1. What are the compute and memory costs of per-layer SVDs and projector construction?
2. Could the projector be adapted online without destabilizing the already-trained adapters?
3. How many harmful prompts are needed before the projector meaningfully stabilizes safety metrics?
4. Does the projector reduce model helpfulness on borderline safety queries that require nuanced refusals?

### Soundness
3

### Presentation
4

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
This paper, "A GUARDRAIL FOR SAFETY PRESERVATION: WHEN SAFETY-SENSITIVE SUBSPACE MEETS HARMFUL-RESISTANT NULL-SPACE" (GuardSpace), addresses the critical problem of safety degradation in pre-aligned Large Language Models (LLMs) during low-rank adaptation (LoRA) for downstream tasks. The authors propose a novel guardrail framework composed of two key components:
1. Safety-Sensitive Subspace Initialization: This method explicitly decomposes pre-trained weights into safety-relevant and safety-irrelevant components using covariance-preconditioned Singular Value Decomposition (SVD). Low-rank adapters are then initialized from the safety-irrelevant components, while the safety-relevant parts are frozen to preserve the original safety mechanism.
2. Harmful-Resistant Null-Space Optimization: A null-space projector (P) is constructed based on the covariance matrix of activations elicited by safety-triggering prompts. This projector constrains adapter updates during fine-tuning, ensuring that the updated weights do not alter the safe outputs on harmful inputs, thereby maintaining the original refusal behavior.
Experiments across various models (Llama-2-7B-Chat, Qwen-2-7B-Instruct, Gemma-2-9B-IT) and tasks (SST-2, AGNEWS, GSM8K) demonstrate that GuardSpace achieves superior performance, notably reducing the average harmful score (HS) and improving fine-tuning accuracy (FA) compared to state-of-the-art baselines like AsFT. The ablation study confirms the crucial role of both components, particularly the null-space projector, in achieving this favorable safety-utility trade-off.

### Strengths
*Originality & Clarity: The GuardSpace framework is highly original, introducing a principled, two-pronged approach to safety preservation during LoRA. The concepts of safety-sensitive subspace (via covariance-preconditioned SVD) and harmful-resistant null space are clearly articulated and technically well-motivated.

*Quality & Significance of Results: The empirical results are compelling and significant, showing a clear and consistent superiority over state-of-the-art safe fine-tuning methods like AsFT across various models and tasks. Achieving near base-model safety (HS) while maintaining or improving fine-tuning accuracy (FA) addresses a crucial limitation of current adaptation techniques.

*Ablation Study: The ablation study is excellent, clearly demonstrating that the null-space projector is the primary driver of safety preservation, while the subspace initialization provides complementary benefits and an effective starting point for task learning

### Weaknesses
1. Computational/Data Overhead: The method requires calculating the covariance matrix C and its SVD/Inverse (C^{−1}) for each linear layer of the LLM before fine-tuning, which can be computationally intensive and demands a dataset of "safety-triggering prompts" a priori. Although the paper mentions the sampling dataset, the overhead of this pre-computation step compared to standard zero-initialized LoRA is not quantified.

2. Invertibility of C: The method for ensuring the invertibility of C by adaptively adding positive values to its diagonal  is practical but mathematically an approximation. A more formal analysis or empirical justification of how this adaptive regularization impacts the fidelity of the safety-sensitive subspace decomposition would strengthen the method.

3. Generalizability of Null Space: While the paper shows the null space generalizes well to different harmful corpora (Fig. 2a) and to unseen harmful data is expected , the key to the harmful-resistant null space is C = XX^T  from the initial safety-triggering prompts H. The long-term stability and generalization capacity of this fixed null-space projector P on completely novel, out-of-distribution jailbreak attacks (post-fine-tuning) remains an open question that should be acknowledged or briefly discussed.

### Questions
1. Complexity and Computational Cost: Could the authors provide an estimate of the pre-computation time (for SVD of WC and SVD(C) for all layers) and memory overhead required for GuardSpace compared to a standard, zero-initialized LoRA setup on a typical Llama-2-7B-Chat configuration? This would help in assessing the practical deployability of the method.

2. Adaptive Inversion of C (Eq. 4): The process for adaptive inversion of C involves adding a regularizing term to the diagonal until l_2 distance between CC^{−1}  and I is small. Can the authors provide the specific positive scaling factor used in the experiments? Additionally, could you comment on whether a simpler, fixed-small-value regularization (e.g., ϵI) would suffice, and why the adaptive approach was chosen?

3. Alternative Safety-Relevant Extraction: The paper relies on covariance-preconditioned SVD to define the safety-relevant/irrelevant subspace. Have the authors considered or experimented with simpler, computationally lighter alternatives for defining the low-relevance subspace, perhaps based on a spectral gap in the singular values of plain SVD(W) or SVD(C), and how these alternatives compare to the proposed method in terms of the resulting safety/utility trade-off?

### Soundness
4

### Presentation
3

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
Many recent works have shown that fine-tuning an LLM degrades its safety alignment. Fine tuning on even a handful number of harmful samples has shown to degrade safety alignment. The paper proposes a fine-tuning-stage defense for mitigating safety degradation due to fine-tuning, primarily focusing on LoRA fine-tuning. The first step is to identify 'safety-relevant' and 'safety-irrelevant' components for each linear layer. Low rank adapters are initialized with the 'safety-irrelevant' components. Next step is to compute a projection matrix that projects into the same null space as that of the activations of harmful inputs. Experiments are primarily focused on fine-tuning Llama-2-7B-Chat on GSM8k. Table 1 considers 2 other fine-tuning datasets (SST-2 and AGNews) and Table 2 considers 2 other models (Gemma-2-7B-IT and Qwen-2-7b-IT) on GSM8k.

### Strengths
* The problem of fine-tuning degrading safety alignment is practically relevant and has recently received a widespread attention.
* The paper is generally well-written.

### Weaknesses
* Experiments are limited, focusing on tuning Llama-2-7B-Chat on GSM8k. It is not clear if the results generalize to other models when fine-tuning on other datasets (SST-2 and AGNews). Ablation experiment in Table 4 is also restricted to tuning Llama-2-7B-Chat on GSM8k. It will be helpful to conduct experiments with more combinations of models and datasets to test generalization of the results. 
* Furthermore, LoRA experiments consider only query and key projections as target modules. It is not clear if the results generalize when additional/different target modules are used. LoRA rank is fixed to 8 and alpha value does not seem to be specified. It will be important to conduct experiments on a variety of LoRA parameters to assess the impact of the proposed method.
* While the paper compares with multiple baselines, only AsFT [Yang et al., 2025] falls into the category of fine-tuning-stage defense. It is a bit unfair to compare the proposed fine-tuning-stage method against post-fine-tuning-stage methods and alignment-stage methods. (Post-fine-tuning defenses as well as defenses such as SafeInstruct allow one to use training libraries such as Transformers PEFT out of the box, whereas fine-tuning stage defenses such as the proposed one require changes in tuning libraries.) It will be helpful to compare the defense against strong fine-tuning-stage defenses such as [Qi et al., 2025]. Note that even though [Qi et al., 2025] focuses on full fine-tuning, the key idea is the modified loss function which can be trained with LoRA as well.

References:
1. Qi et al., "Safety Alignment Should be Made More Than Just a Few Tokens Deep", ICLR 2025

### Questions
* The results for Llama-2-7B-Chat in Table 1 match with the column of $p = 0.10$ in Table 3. Does Table 1 consider adding 10% harmful prompts in GSM8k? It is important to give more details. 
* It will be helpful to expand on the notion of 'unsafe proportion' in Line 403 (it is clear to readers familiar with the topic, but other readers may not be aware). Also, which dataset is used for unsafe prompts? 
* Algorithm 1 mentions to perform forward propagation using $W' + BAP$, whereas eq (8) defines $W' = W - BAP$. This essentially suggests that forward propagation is done with $W$, which is confusing. Can the authors provide more details?
* All the models used in the paper are fairly old, especially Llama-2-7b-Chat (considering the fast pace of LLMs). Are there any experiments on any of the newer versions of the models (e.g., Llama-3-8b-IT, or Gemma-3-1b-IT)?
* Including 'guardrail' in the title can be quite confusing, since safety 'guardrails' typically indicate external tools/methods used to preserve safety. Can the authors provide more details on referring the proposed method as a 'guardrail'?

### Soundness
2

### Presentation
3

### Contribution
2
