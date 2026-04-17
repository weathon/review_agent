# At the Edge of Understanding: Sparse Autoencoders Trace The Limits of Transformer Generalization

- Decision: Reject
- Scores: 6, 2, 8, 6, 2

## Abstract
Pre-trained transformers have demonstrated remarkable generalization abilities, at times extending beyond the scope of their training data. Yet, real-world deployments often face unexpected or adversarial data that diverges from training data distributions. Without explicit mechanisms for handling such shifts, model reliability and safety degrade, urging more disciplined  study of out-of-distribution (OOD) settings for transformers. By systematic experiments, we present a mechanistic framework for delineating the precise contours of transformer model robustness. We find that OOD inputs, including subtle typos and jailbreak prompts, drive language models to operate on an increased number of fallacious concepts in their internals. We leverage this device to quantify and understand the degree of distributional shift in prompts, enabling a mechanistically grounded fine-tuning strategy to robustify LLMs. Expanding the very notion of OOD from input data to a model’s private computational processes—a new transformer diagnostic at inference time—is a critical step toward making AI systems safe for deployment across science, business, and government.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a framework based on SAE to analyze the generalization of LLMs to OOD inputs. The authors first analyze how OOD inputs can lead to unusual features or spurious concepts within the model, which can be observed using SAE tools. They then experimentally demonstrate that typo inputs degrade model performance and that SAE-derived indicators can capture input distribution shifts. Using SAE in the jailbreak scenario, the authors observe a similar distribution shift. Finally, they demonstrate that fine-tuning using SAE-derived indicators improves the model's robustness to typo inputs and jailbreak attacks.

### Strengths
1. This work is comprehensive. It includes experiments on synthetic tasks, fine-tuning, and application to more realistic jailbreak scenarios, fully demonstrating the authors' claims.
2. At least in my opinion, using the SAE indicators to observe the input distribution shift and fine-tune the model based on that is novel.
3. The results on models of varying sizes and different tasks make their conclusions have certain practical value.

### Weaknesses
1. Figure references are unclear. Figure 1 and Figure 2 lack necessary references. Figure 4A has an incorrect reference.
2. The conclusions in Sections 4.1 and 4.2 are not surprising. It seems intuitive that typo inputs affect model performance and have a positive correlation.
3. The experimental design in Section 5 is reasonable, but the experiments are conducted on GPT2. Fine-tuning on a larger model, such as llama, can better support the conclusions.

### Questions
1. Have you tried fine-tuning on a larger model (e.g. llama) to support the conclusions in Section 5, even using LoRA? Plotting a figure similar to Figure 2a before and after fine-tuning would be a good way to support the conclusions.
2. The authors study character-level tokenization, which is not the mainstream tokenization granularity of current LLMs. Will it affect the conclusions of the paper?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work focuses on using mechanistic interpretability method to detect OOD data/model jailbreak attempts. 

1. The study proposes using sparse autoencoders (SAEs) as a mechanistic framework to analyze transformer residual stream activations. It finds that out-of-distribution (OOD) inputs, such as subtle typos and adversarial prompts, cause the model to activate a significantly larger number of spurious internal "concepts" compared to in-distribution data.
2. This internal off-manifold activation is directly correlated with degraded external performance. Introducing typos into MMLU benchmark questions caused significant accuracy drops in models ranging from Llama 3.1 8B to GPT-5-thinking-nano .
3. The framework is applied to make models more robust: an SAE-derived "energy score" is used to select the most OOD samples for more efficient fine-tuning, and the method identifies successful jailbreaks as OOD (activating more exclusive features). A targeted LoRA fine-tuning subsequently aligned these internal activations, reducing the jailbreak success rate.

While the paper presents empirically effective results, it lacks a comprehensive scientific framework that explains how these SAE-based methods can be applied by other practitioners and generalized beyond the specific models and data used in the study.

### Strengths
1. The typo-based experimental design for the toy-problem is well-crafted.
2. SAE-based methods demonstrate their ability to detect out-of-distribution data and identify jailbreak attempts. 
3. The proposed methods for enhancing fine-tuning efficiency and mitigating jailbreaks are highly effective.

### Weaknesses
1. The paper lacks effective guidelines for utilizing raw energy scores or the number of activated concepts to determine if data is out of domain. Instead, it relies solely on simple comparisons. Statistical methods and guidelines, such as the number of standard deviations from the mean, would be more easily applicable.
2. The paper explores the application of SAEs at various network layers, but it lacks sufficient guidelines for selecting a specific layer when applying the method.
3. Limitations are not explicitly mentioned.

### Questions
1. What are the main limitations of the work?
2. Can you provide guidelines for using the energy score and the number of activated features to identify jailbreaks?
3. Could you please provide your default recommendation for the layer to use when applying your method? Additionally, could you explain the rationale behind this choice?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper is a mechanistic study on the ability of LLMs to generalise to OOD, with a specific focus on controllable generation of datapoints and an emphasis on lexical over semantic perturbations. The methodology operates under the manifold hypothesis, which allows for the definition of intuitive metrics to explain robustness. The topic and thesis are not strictly novel: work on transformer robustness, especially to typos, has existed since at least 2020 with the same takeaways. However, this type of work has (surprisingly) not been explored in LLMs. This contribution, added to the robust, systematic approach to experimentation and solutions proposed, makes this paper a strong contribution to the field.

### Strengths
In my opinion, this paper is very strong. It is thorough, well-written, and has a good, robust, set of experiments. All around, save for a few comments (in 'Weaknesses'), it makes a good contribution to the study of LLM generalisation. Here are a few of the more salient things of this work:
1. A large set of experiments examining the central thesis of the work _without_ overlaps (i.e., a main evaluation and a set of complete ablation studies). 
2. Use of open/closed models with appropriate versioning and disclosures/releases. Also use of non-pretrained and pretrained models alike.
3. Experiments have a rigorous backing and well-defined metrics. In particular, the authors display a good understanding of the behaviours and particularities of SAEs / OOD data generation and use these to their advantage to provide more robust results.
4. Well-written paper, with some minor comments (below). In particular, it clearly states contributions, framing, and limitations.
5. Contributions go beyond pointing out issues with LLMs, and show the applicability of SAEs for robustness improvement, especially on jailbreaks.

### Weaknesses
I have a few minor comments.

1. Better separate results from discussion. Every experiment (and sub-experiment) has results and then discussion/conclusion attached. However, they become conflated. E.g., L369 is an interpretation of the results from Sec 5 _and_ Sec 4. But Sec 5 already is working on assumptions drawn from the interpretation of the results from Sec 4 (L306 -- 'The results in Section 4 present evidence...'). The (interpretation) of the results from Sec 4 is given in both 4.1 and 4.2. This structure could be misleading and does raise questions on the rigour (i.e., objectivity) of the interpretation of the experiments.
    - This is _especially_ important since the approach to the study of LLMs is a mechanistic analysis. In other words, the work implicitly relies on causal relationships between the hypotheses and the data. Thus, clearly separating what it is observed from what follows (as an interpretation), is necessary to showcase sufficient rigour. 
    - As an example of this: the motivation of Sec 5 follows from Sec 4, but there are likely alternative explanations to the observations from 4.1 and 4.2 (see the questions below).
2. Reproducibility is definitely there, but I would suggest also adding the versions (or dates) for the calls to proprietary LLMs. They tend to be updated quite frequently.
3. Formality of the writing needs to improve in some (very minor!) areas: namely, while English morphology is quite flexible, I would argue that 'robustify' is a bit less formal than desired for a scientific publication. Another example would be L412: the '=' sign inline for text shouldn't be used here (this isn't notes, it is a final, publication-worthy work).
4. Also very minor: it's 'GPT-2', not 'GPT2'. Watch out for the citation in L142

### Questions
MMLU is known to be contaminated, especially in Llama / GPT models. How much do you think this influenced the observations from 4.2, and their relationship to 4.1, where it is argued that TinyStories is not memorised? The conclusion from 4.2 is that 'reasoning' is impacted by typos, but couldn't also be simply by less effective string matching?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper investigates how sparse autoencoders (SAEs) can be used to identify and understand out-of-distribution (OOD) behavior in transformers. The authors showed that OOD inputs (including typos and jailbreak prompts) cause LLMs to generate spurious concepts in their internal representations, which can be detected by sparse autoencoders. They further extended this observation to achieve the following tasks: (1) quantifying distributional shift, (2) designing more efficient fine-tuning by selecting high-value OOD samples, and (3) defending against jailbreak attacks through SAE-informed alignment.

### Strengths
1. The problem being investigated is critical. Effectiveness within the constraint of OOD has been a long but unresolved problem.

2. The application of SAE is novel and interesting. SAE is used to diagnose OOD behavior in a transformer's interior, which is an extension to traditional OOD detecting method.

3. The empirical study is extensive and convincing. Experiments span over multiple scales, modalities (vision and text), and model levels (toy models to frontier models).

4. The energy score metric is novel and effectively combines reconstruction error and spurious concept activation.

### Weaknesses
The energy score formula could be better motivated before introduction. This metric, also effectively from the empirical lens, seems ad hoc. Additional explanations on this metric would be very much appreciated.

### Questions
Please see Weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors aim to analyze the OOD behaviour in language models using sparse autoencoders. For this, they train sparse autoencoders on a trained model and then corrupt the input samples using typos and test the features that gets activated in SAEs. They find that on using such perturbations, several features get activated as compared to using in-distribution samples without corruptions. They further analyze that using these corruptions the MMLU performance of llms is degraded and hypothesize the presence of large number of features getting activated in SAEs being the cause for that. They further fine-tune the models on corrupted input samples and observe that this reduces the activations in SAEs, demonstrating that the larger number of SAE activations are closely related with OOD detection. They further use this to detect jailbreaks in the Llama-3.1 8B model.

### Strengths
* Using SAE to perform OOD detection is an interesting direction. This can help in detecting some adversarial samples as well.

### Weaknesses
* The experiments performed in this paper define corruptions in the input tokens as OOD samples. However, this is a very small subset of possible OOD samples. Therefore, authors would need to scale up their experiments and evaluations to demonstrate that SAEs can really help with OOD detection. 
* SAEs are known to be less reliable as training them on different samples can lead to extraction of different features [1,2 ]. One way to resolve this issue to some extent could be to train the SAEs on a very large number of samples. However, in these cases as well it might be possible to craft some adversarial attacks which can evade being detected by SAEs.

[1] Projecting Assumptions: The Duality Between Sparse Autoencoders and Concept Geometry (https://arxiv.org/abs/2503.01822)

[2] Analyzing (In)Abilities of SAEs via Formal Languages (https://arxiv.org/abs/2410.11767)

### Questions
It would be great if the authors can perform more extensive evaluation to demonstrate the robustness of their claims. For example, they can analyze different kinds of OOD samples instead of mere input corruptions. They can include change in format or style of the samples as compared to the ones used for training SAEs.

### Soundness
2

### Presentation
2

### Contribution
1
