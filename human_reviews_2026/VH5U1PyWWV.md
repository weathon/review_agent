# SAEdit: Token-level control for continuous image editing via Sparse AutoEncoder

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Large-scale text-to-image diffusion models have become the backbone of modern image editing, yet text prompts alone do not offer adequate control over the editing process. Two properties are especially desirable: disentanglement, where changing one attribute does not unintentionally alter others, and continuous control, where the strength of an edit can be smoothly adjusted. We introduce a method for disentangled and continuous editing through token-level manipulation of text embeddings. The edits are applied by manipulating the embeddings along carefully chosen directions, which control the strength of the target attribute. To identify such directions, we employ a Sparse Autoencoder (SAE), whose sparse latent space exposes semantically isolated dimensions. Our method operates directly on text embeddings without modifying the diffusion process, making it model agnostic and broadly applicable to various image synthesis backbones. Experiments show that it enables intuitive and efficient manipulations with continuous control across diverse attributes and domains.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces SAEdit, a novel method for achieving continuous and disentangled image editing by manipulating text embeddings at the token level. The core idea is to train a Sparse Autoencoder (SAE) on the output embeddings of a frozen T5 text encoder. This SAE maps the embeddings into a high-dimensional, sparse latent space where semantic attributes are isolated.

The primary contributions are: 1) A new framework that enables continuous and disentangled editing in diffusion models. 2) A model-agnostic approach that operates purely on text embeddings, making it compatible with various T5-based backbones (demonstrated on Flux and Stable Diffusion 3.5) without retraining. 3) Strong qualitative and quantitative results, evaluated via LPIPS, a VQA-based score, and a user study, which show it outperforms existing methods.

### Strengths
This paper introduces SAEdit, a novel and effective method for continuous and disentangled image editing by manipulating text embeddings. Strengths are:
1. Good Novelty: The central idea of applying an SAE to the text embedding space to discover disentangled semantic directions is highly innovative. Previously, GAN was much better in terms of such continuous editing ability. This paper introduces SAEdit, which enables diffusion models to have the same ability and can be applied to general methods.

2. Great Disentanglement and Control: The paper effectively addresses two major challenges in image editing: disentanglement and continuous control. The qualitative results clearly show that SAEdit can do continuous target edit without changing the objective ID.

3. Generality: A key advantage of SAEdit is its model-agnostic nature. By operating only on the text embeddings, it can be seamlessly applied to different T5-based text-to-image backbones without any model-specific retraining. 

4. Well Written and Organized: The paper is well-written, clearly structured, and easy to follow. The methodology is explained in detail, and the figures are highly illustrative.

### Weaknesses
1. Because of the sparse learned space, the edit directions actually cannot be determined with simple words. So this would rely on LLM to generate multiple prompts, which can enrich the edit directions, make it clearer. But maybe that's why most image editing works would have a reference image. Wonder if authors can use a reference image to guide the edit directions. Image would provide much enriched direction information than text.

2. Potential Need for Manual Refinement: The limitations section notes that for some complex edits, "manually selecting or de-selecting a few specific entries in the sparse direction vector can yield even more disentangled results." This suggests that the fully automated direction-finding process is not always optimal and may require human intervention for the best performance, which slightly reduces the method's practical autonomy.

3. As the authors rightly point out, the method is constrained by the biases and semantic entanglements of the underlying text-to-image model. The failure cases on OOD edits are a clear example. While this is a shared limitation for many editing methods, it's a key boundary on the method's capabilities.

4. Evaluation Metrics are not that clear. I understand this paper's task is hard to quantify, so they only use the VQA score and LPIPS. The most desired case is that authors also introduce an evaluation benchmark to quantify the results.

### Questions
The main concern is the evaluation. If authors can introduce a more valid evaluation way or explain thoroughly, I will increase my score.

Also, there exists a little concern about using LLM to enrich the edit direction. But I think this one might hard to deal with, because of the sparsity nature of text, compared with image.

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
5

### Summary
The paper proposes a Sparse AutoEncoder to learn sparse representations of attributes for disentangled edits and continuous control in diffusion models. It identifies a robust edit direction by selecting the largest singular vector of the matrix of N steering vectors using N different prompts. The method does not require task-specific optimization and only uses text embeddings for manipulating the attributes. Experiments are shown on SD3.5 and Flux diffusion models using LPIPS and VQA score metrics along with a user study.

### Strengths
The paper is clearly written and easy to read.

The proposed text based attribute control is scalable as it can be applied to all model architectures sharing the same text encoder.

It does not require task specific optimization.

### Weaknesses
Text-based attribute control is not new as prior works such as Attribute Control or PromptSliders have explored this idea. The proposed SAE based attribute control method is also not new since it is just applying to this task and so novelty is limited.

The  proposed method is not thoroughly evaluated against existing methods. Qualitative results comparing to Attribute-Control and Prompt Sliders are missing. Attribute Control can also applied for disentangled edits shown in Figure 14, 15 and 16.

Only two attributes are shown for composing edits. Does it work with multiple concepts together? The claim of disentanglement is not properly justified although the method is motivated by this property. Quantitative evaluation of composing multiple attributes should be conducted to justify as only few qualitative results are not sufficient to prove its robustness.

Some of the attribute edits are fundamentally limited by the semantic representation learned by the text encoder. For example, Figure 2 only changes the beard without changing the hair or face for being old. 

Quantitative results on Face-ID is missing. Some of the edits sometimes changes the identity of the persons, e.g., Figure 6 blonde and smile examples. Figure 13 second row.

How sensitive is it to the seed of the diffusion model? Is the performance consistent across different seeds?


What about the concepts not well represented by the diffusion model? Unlike methods like concept sliders that can learn from images, this is a fundamental limitation of text based slider methods.

### Questions
See the weaknesses above.


Implementation details missing. How many concepts or edit attributes were extracted? There are only a few qualitative results shown for few concepts. Is the method robust to different attributes or styles?

How does the method work with CLIP text encoder?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper employs sparse autoencoders on text embeddings to obtain disentangled semantic representations, so as to enable continuous attribute-specific image editing. Experiments show the effectiveness of the proposed method in providing disentangled and continuous image edits.

### Strengths
1.	The idea of employing sparse autoencoder to obtain disentangled semantic representation from text embedding is reasonable. 
2.	The visual results of image editing in the empirical study are promising.

### Weaknesses
1.	Extracting semantic directions from text has been explored in [1] but is not discussed in this work. A detailed empirical comparison between the newly proposed method and [1] is lacking, which makes it difficult to judge the advancement of this work over [1].
2.	Several recent works on text-to-image editing (e.g., [2–4]) are missing from the related work. These studies demonstrate disentangled and continuous editing, so comparing with them is necessary to better show the proposed method’s advantage.
3.	The paper should also discuss related works on applying sparse autoencoders for disentangled representation learning [4]. In particular, several recent studies [5,6] have already employed sparse autoencoders in text-to-image models. Therefore, the proposed idea appears not to be novel, and the contribution of this work is marginal.

### Questions
1.	Is the relevant source token manually determined? If so, the approach seems too heuristic and labor-intensive, as it requires identifying the token for each given source prompt.
2.	I wonder whether the hyperparameter $\rho$ is sensitive. It would be helpful to include a sensitivity analysis of this hyperparameter.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The authors propose SAEdit, which aims to obtain disentangled and controllable editing directions to perform semantic edits on generated images. To achieve this task, the proposed method trains sparse autoencoders (SAEs) for source and target prompt pairs, using the embeddings obtained from the text encoder of the designated diffusion model (T5-XXL and FLUX, respectively). Over the provided qualitative and quantitative analyses, the method shows its effectiveness to be able to perform edits in a continuous way while preserving the disentanglement properties, with directions applied to the corresponding subject tokens.

### Strengths
- The method shows strong performance in terms of the continuity of the edits compared to the baselines. As summarized in Fig. 10, SAEdit achieves a continuous editing direction where the linearity of the effect can be quantified with the change in VQAScore.
- Editing directions trained with SAEdit do not require any extensive data collection for input-edit pairs, but operates with easily collectable prompt pairs.
- The method shows satisfactory performance over editing cases where the intended edit is ambiguous, which is challenging for the majority of the editing methods (e.g. surprised), as well as cross-domain.
- The quantitative experiments and user studies demonstrate the effectiveness of the method.

### Weaknesses
- The examples included in the paper are mainly human-centric examples, which may also be the case in the quantitative evaluations. The authors are strongly encouraged to provide more details on their evaluation setup and how diverse these 63 samples are.
- The examples for real image editing are limited, it should be clarified if the selective editing scheme is also valid for real image editing cases.
-  The limitations section can be extended with what are the types of edits that the method cannot tackle, if there are not such limitations, authors are encouraged to include more diverse examples for editing cases.

### Questions
- Does the proposed method require training of SAEs per generation prompt? From Fig. 5, it seems like the trained SAE is different for every generation prompt. If this is the case, what is the computational overhead that is required for every generation? For the robustness enhancement with SVD, with how much deviation training a new direction would be required?
- What are the generalization properties of the trained. SAE representations and their applicability to similar but slightly different prompts? As an example, if we train a SAE for the prompt pair ("a woman", "a smiling woman"), is this applicable to images generated with "a man" (to transform it into "a smiling man")?
- How does the method perform in stylization scenarios? Does the trained sliders handle gradual style changes?

### Soundness
3

### Presentation
3

### Contribution
3
