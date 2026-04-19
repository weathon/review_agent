# Concept Bottleneck Large Language Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 5

## Abstract
We introduce Concept Bottleneck Large Language Models (CB-LLMs), a novel framework for building inherently interpretable Large Language Models (LLMs). In contrast to traditional black-box LLMs that rely on limited post-hoc interpretations, CB-LLMs integrate intrinsic interpretability directly into the LLMs -- allowing accurate explanations with scalability and transparency. We build CB-LLMs for two essential NLP tasks: text classification and text generation. In text classification, CB-LLMs is competitive with, and at times outperforms, traditional black-box models while providing explicit and interpretable reasoning. For the more challenging task of text generation, interpretable neurons in CB-LLMs enable precise concept detection, controlled generation, and safer outputs. The embedded interpretability empowers users to transparently identify harmful content, steer model behavior, and unlearn undesired concepts -- significantly enhancing the safety, reliability, and trustworthiness of LLMs, which are critical capabilities notably absent in existing language models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes the Concept Bottleneck Large Language Model (CB-LLM), a novel approach to interpreting language model (LM) decisions by connecting its hidden states to defined concepts. The CB-LLM algorithms for both text classification and text generation are thoroughly detailed and well justified. Overall, the proposed methods are both interesting and effective.

### Strengths
1. The idea is interesting and novel. The motivations, research gaps, and connections to prior studies are well articulated.

2. The authors explore and validate CB-LLM in both text classification and text generation contexts. Each step’s details, challenges, and solutions are clearly illustrated.

3. Extensive experiments, visualizations, and case studies effectively demonstrate the proposed method’s effectiveness.

### Weaknesses
1. The concept space for text generation explanations seems only limited to class labels. Specifically, the explanations tend to indicate the model’s generation direction (e.g., positive or negative sentiment) but remain constrained to class labels. Recent studies [1,2] suggest that explaining LLM outputs in generation contexts should extend across the full vocabulary. For instance, contrastive explanations can reveal why a model generates one token over all other possible alternative tokens in the whole vocabulary, which offers a more fine-grained analysis than CB-LLM currently provides.

2. The experimental setup in Section 4.2 lacks clarity. It would help to specify the input prompts used for measuring accuracy, steerability, and perplexity, as well as the length of generated texts.

3. Steerability is insufficiently compared. First, the prompts used for each class are unclear and under-detailed. Additionally, steerability should be evaluated against the original generations without any control mechanisms for a baseline comparison.

4. I am curious about the diversity of generated sentences using CB-LLM. Could the authors add experiments to assess diversity, such as generating multiple outputs from the same prompt and then evaluating variation across those outputs?

5.	I recommend that the authors refer to recent work on controllable text generation such as [3] to enhance the experimental design and presentation of results, which may help address weaknesses 2–4.

6.	Minor points: The full name of CAV in Section 2 is not provided. Also, the word "We" in the first sentence of Section 4 should start with a lowercase letter.

References:

[1] Interpreting Language Models with Contrastive Explanations, EMNLP 2022

[2] Unveiling and Manipulating Prompt Influence in Large Language Models, ICLR 2024

[3] Controllable Text Generation via Probability Density Estimation in the Latent Space, ACL 2023

### Questions
1. What is the agreement coefficient (like Kappa coefficient) for the work evaluations in Tables 3 and 4?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes to add concept bottleneck layers (CBLs) to LLMs (right before softmax layers for text-classification, and softmax layers for token generation). Each output neuron of a CBL represents a concept (e.g. "sport", "car"). This will increase interpretability of LLMs: the concept bottlenecks will tell possible concepts of text label (in text classification), and possible concepts of the context to generate a token. Different than previous works, although the concepts learnt are also given by GPT, they are generated only once by asking GPT to generate concepts for each class (rather than for each instance as in previous work). Thus reduce GPT query cost. Although CBLs have been used in computer vision, the paper claims that this is the first work employing CBLs for LLMs.

The paper shows that the CBL used for text classification and text generation doesn't drop the performance of the used LLM backbone. Yet, it improves interpretability. Specifically, it can help to manipulate LLM's generated text.

### Strengths
The text generation part is simple but interesting.

### Weaknesses
The first part of the paper presents how CBL is used in text classification. I found this part unsatisfying. 
* The core idea actually is to learn a mapping between concepts generated by GPT and the label set, and thus it seems to me that this part can be explained easily in a much shorter text. Especially, the whole idea of step 3 (correction), written in more than half a page, is simply to zero out those concepts that aren't generated by GPT given a target label -- I believe we just need a paragraph for that. 
* What struggles me is the claim of improving interpretability. In fact, what happens here is that, if a text x is mapped to a label y via concepts Z, we can "explain" that label y is chosen because concepts Z are chosen. But that is because of the way GPT generating concepts for label y, and the way the mapping is learnt. It doesn't really explain how and why concepts Z are chosen for text x. What I see here is that the method simply replaces the text classification on the original label sets to the new sets of concepts. 
* The "unlearning" section is a bit untraditional to me. Removing a subset of concepts doesn't seem to me "unlearning" as the model parameters are still unchanged. 
* I found the experiments quite not adequate as only one LM is used. Would be better to see a wider range of backbone LMs here.  

The second part, text generation, is interesting to me. However,
* the paper criticises C3M because it requires human-annotated concepts. But the text generation actually requires human-annotated concepts too (e.g. see fig 3). So it seems to me that criticism is unfair.
* I think this part is related to the controllable text generation in the literature. I would like to see a discussion about the relationship.
* I think the experiments should include some baseline in controllable text generation literature. 
* The interpretability analyses should include some analyses about the meaning of the latent concept neurons and their impacts to the final generated text.

### Questions
* What is S_c in equation 2?
* Sec 3, why not use CEBaB & IMDB as in C3M paper?
* Which parameters are optimized in equation 5? 
* What is concept accuracy in sec 4.2?
* In "generation quality" sec 4.2, sentence "If the generated sentence lacks fluency, the perplexity can rapidly rise to around 500. Therefore, a small difference in perplexity would not affect the generation quality." --> how can you quantify it? what small is "small"?

### Soundness
3

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
3

### Summary
The paper presents CB-LLM, designed to provide inherent interpretability for both text classification and text generation tasks. In text classification, the model performs comparably to traditional black-box models while offering interpretable reasoning. For text generation, the model introduces interpretable neurons that enable concept detection and controllable text creation, presenting potential applications such as chatbot toxicity moderation.

### Strengths
- The paper presents a novel method for efficient interpretation in text classification tasks and proposes the first method for interpretation in text generation tasks.
- Both methods have somewhat practical applications for real-world tasks.
- The paper is easy to follow and includes rich figures and examples.

### Weaknesses
1. There is limited discussion about the generalizability of interpretable layers across various domains or tasks, specifically regarding whether concepts learned in classification or generation from one dataset can be reused in another. This lack of focus on transfer learning limits the model's broader applicability to related but unseen tasks. Moreover, the method is only tested on LLaMA-3-8B, so it is also important to investigate whether it consistently works across different model families and sizes.
1. It remains uncertain if the generated concepts consistently align with human intuition across all datasets.
1. The reliance on ChatGPT and sentence embedding models raises concerns about performance variability. Given their close performance levels, choosing different models or conducting multiple experiments could help mitigate this variability.

### Questions
1. With reference to Section 3.2, I am uncertain about the precise number of trained models: RoBERTa, GPT-2, and LLaMA-3. Could the authors succinctly clarify the training process and the employed LM, LLM, and sentence embedder?
2. In Figure 1, should the value prior to correction be **-** 0.1?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces the Concept Bottleneck Large Language Model (CB-LLM), a novel large language model (LLM) that offers clear and accurate explanations through built-in interpretability and scalability, representing a breakthrough compared to traditional black-box LLMs. CB-LLM not only narrows the performance gap with black-box models in text classification tasks but also demonstrates how to leverage interpretable neurons for concept detection and guided text generation in text generation tasks. The authors conducted experiments across multiple datasets and metrics, showing that CB-LLM achieves considerable performance while maintaining inherent interpretability compared to black-box models. Additionally, the authors point out that CB-LLM not only possesses inherent interpretability during text generation but also provides greater controllability over the generated content. This feature gives CB-LLM potential application value in various fields requiring controllable content generation.

### Strengths
1. CB-LLM demonstrates outstanding performance in text classification tasks. After the introduction of Automatic Concept Correction (ACC), the model surpasses traditional black-box models on several datasets, showcasing its strong potential.
2. During the training process for text classification, CB-LLM effectively removes negative activation values using the ReLU activation function, significantly reducing semantic ambiguity during training and enhancing model stability.
3. The authors provide evidence through human evaluation that CB-LLM received positive ratings for both Activation Faithfulness and Contribution Faithfulness, indicating that the model's explanations align more closely with human understanding and expectations, thereby enhancing the model's transparency and trustworthiness.

### Weaknesses
1. In the CB-LLM for text classification , the introduction of sparsity constraints during training is not adequately explained. If this constraint aims to enhance the model's interpretability, the authors should provide more analysis and experimental evidence to support this claim.
2. The authors used only RoBERTa and GPT-2 as backbone models in their experiments. Given that the main goal of the paper is to improve the internal interpretability of large language models (LLMs) in text classification, it would be beneficial to conduct related experiments using larger models, rather than relying solely on LLaMA for labeling.
3. Although CB-LLM applies the concept bottleneck to text generation tasks, there is a lack of sufficient experimental validation regarding its internal interpretability when generating text. In section 4.2, the “concept detection” experiment is conducted only on a text classification dataset, which does not effectively demonstrate whether CB-LLM's internal interpretability in text generation is faithful and valid.
4. Some statements in the paper may need to be revised for clarity and to avoid ambiguity. For example, the variable “y” in Equation 2 lacks sufficient contextual explanation. Additionally, both Table 2 and Table 6 mention “RoBERTa-base fine-tuned (black-box),” but the values differ. The “RoBERTa-base fine-tuned (black-box)” in Table 6 should be replaced with “GPT-2” to eliminate confusion.

### Questions
1. I find the illustration in Figure 3 somewhat confusing. Module 1 is represented in green text; does this mean that all green parts belong to Module 1? Similarly, do the blue parts represent Module 2? Due to the positioning of the text, I have difficulty distinguishing between the two modules. If my understanding is correct, I recommend adding some identifiers to clarify this distinction.
2. I found it difficult to understand the concept of "concept bottleneck", which is widely used in the field of computer vision. I hope that a brief description of the concept bottleneck can be included in the related work section, as this would help readers from unrelated fields to quickly understand the paper.

### Soundness
3

### Presentation
2

### Contribution
3
