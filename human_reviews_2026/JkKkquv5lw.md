# A Study on PAVE Specification for Learnware

- Decision: Accept (Poster)
- Scores: 8, 6, 4

## Abstract
``Learnware = Model + Specification''. A learnware comprises a submitted model paired with a specification sketching its capabilities. For a Learnware Dock System (LDS) which accommodates numerous models, these specifications are essential to enabling users to identify helpful models, eliminating the requirement for prohibitively costly per-model evaluations. Recently, Parameter Vector (PAVE) specification, which utilizes the changes in pre-trained model parameters to inherently encode the model capability and task requirements, shows promising capabilities in enabling identifying useful learnwares for high-dimensional, unstructured text data. In this paper, we present a comprehensive study of PAVE specification for learnware identification. Theoretically, from the neural tangent kernel perspective, we establish a tight connection between PAVE and prior specifications, providing a theoretical explanation for their shared underlying principles. We further approximate PAVE in a low-rank space and analyze the approximation error bound, highly reducing the computational and storage overhead. Extensive empirical studies demonstrate that PAVE specification excels at identifying CV and NLP learnwares even from heterogeneous learnware repository with corrupted model quality. Reusing identified learnware to solve user tasks can even outperform user-fine-tuned pre-trained models in data-limited scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces the Parameter Vector (PAVE) specification to identify and select high-quality learnwares for reuse in solving new user ML tasks. The PAVE approach formalizes the use of parameter vector changes (from fine-tuning a shared pre-trained model) as a representation of both a model’s capability and the requirements of the user’s task. Theoretically, the work connects PAVE to kernel mean embedding approaches via the neural tangent kernel regime, derives error bounds for low-rank approximations of the parameter vectors, and demonstrates through extensive empirical studies that PAVE achieves superior learnware selection, often outperforming conventional fine-tuning and previous learnware identification methods.

### Strengths
- The paper tackles the practical challenge of efficiently reusing high-performing models by providing a specification (PAVE) that allows users to identify suitable models without direct per-model evaluation.
- The mathematical exposition is detailed and generally clear.
- The proposal for low-rank approximation of parameter vectors significantly reduces memory and compute, a nontrivial contribution given the scale of modern check-pointed models.
- The experimental validation is exceptionally thorough and is a major strength of the paper.

### Weaknesses
Generally, I did not identify any major limitations or weaknesses in its core contributions. This is a high-quality paper that makes a practical contribution to the field of model reuse and the Learnware paradigm. The proposed PAVE specification is novel, well-motivated, and supported by both theoretical analysis and a comprehensive set of experiments.

### Questions
The PAVE specification, as described in L51-53, appears to rely on a shared pre-trained model and architecture to ensure the comparability of parameter vectors. Could the authors clarify if this is a necessary constraint? Furthermore, could the authors provide discussion on the potential for generalizing PAVE to a more heterogeneous setting, where learnwares in a repository might originate from different base models or possess distinct architectures?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper proposes a new way, called PAVE (Parameter Vector), to describe and identify reusable models in the Learnware framework. Instead of using reduced data samples to represent a model’s capability (as in prior work like RKME), PAVE uses how a model’s parameters change during fine-tuning. This parameter-based “signature” helps match models (learnwares) with user tasks more efficiently and works even when data are high-dimensional (e.g., text or images). The authors also derive a theoretical link between PAVE and RKME through the neural tangent kernel (NTK) and propose a low-rank approximation to make computation feasible. Experiments on NLP and CV tasks show that PAVE can identify suitable learnwares better than baselines and sometimes outperform fine-tuned pre-trained models

### Strengths
1. Using parameter changes instead of data samples to represent a model’s capability is original and intuitive, especially for privacy-sensitive or unstructured data.
2. The connection between PAVE and RKME under the NTK assumption provides a sound theoretical explanation for why the approach should work.
3. The low-rank approximation (similar to LoRA) reduces the storage and computational cost while maintaining similarity accuracy
4. The experiments cover multiple domains (NLP, CV, medical LLMs) and show clear and consistent improvements over RKME and fine-tuning baselines.

### Weaknesses
1. While the paper introduces PAVE as a general solution, it’s not entirely clear how broadly this parameter-vector representation generalizes beyond fine-tuned models based on shared backbones.
2. The method assumes all learnwares are fine-tuned from a common pre-trained model. It’s unclear how it performs when models come from different architectures or pre-training distributions.
3. The experiments mainly compare to RKME and basic fine-tuning baselines. It would strengthen the evaluation to include more data-centric reuse or transferability estimation methods, such as LEEP, LogME, or task2vec, to contextualize PAVE’s advantage.
4. The paper doesn’t discuss how PAVE specifications would be stored, shared, or updated in a real learnware repository, or how they interact with privacy and security constraints.

### Questions
Please refer to the weakness section

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
1

### Summary
The paper considers a setup that we have access to a large number of models. Each model consists of a model and a "specification" that describes its capabilities. The central problem is how to create a specification that allows a user to efficiently identify the most helpful model from a vast repository, without the costly process of evaluating every single model.

The main idea is simple: assume we have access to a trained model h. To create its specification, they take a shared, public pre-trained model (e.g., BERT, CLIP) and fine-tune it to mimic the predictions of their model h. The resulting change in the pre-trained model's parameters is saved as the model vector.  A user with a new task provides a small, few-shot dataset. To create their task specification, they fine-tune the exact same shared pre-trained model on their few-shot data to fit the true labels. This parameter change becomes the task vector. Finally the selection is based on measuring the similarity between task vector and the finetuning vector.

### Strengths
The PAVE method's primary strength is its effectiveness with high-dimensional, unstructured data like images and text, a scenario where prior specifications failed. The authors conducted extensive experiments across Natural Language Processing (NLP),

### Weaknesses
Reliance on a Shared Pre-trained Model: The entire system fundamentally relies on both the developers (creating learnwares) and the users (sketching their tasks) using the exact same shared pre-trained model as a common basis to generate the parameter vectors. I think the authors should study the robustness of their finding s to this assumptions.

### Questions
see above

### Soundness
3

### Presentation
3

### Contribution
3
