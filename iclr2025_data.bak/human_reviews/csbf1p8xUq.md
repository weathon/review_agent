## Human Reviewer 1

### Summary
The paper presents an approach to multilingual machine translation that focuses on mitigating losses on non-high-resource language pairs. The authors focus on 50 languages ranging from low- to high-resource. The approach consists of performing several levels of continued pretraining and post-training using LoRA modules that group language families together. The training recipe introduces a new preference learning method called ARPO that aims to mitigate "over-rejection" of samples that seems to occur in machine translation for pairs of hypotheses that are very similar to each other in which the differences are just a few characters or tokens. Overall the results indicate state-of-the-art results on multilingual MT benchmarks when compared to other open source multilingual LLMs.

### Strengths
The paper is very well written, clear and easy to read. The motivation is solid, timely and important: multilingual LLMs usually do not consistently maintain translation quality when we scale the number of languages. The experimental settings are sound and robust. The preference learning method proposed, ARPO, is very useful and takes into consideration shortcomings of previous methods applied to MT. It is going to be useful for the community working on MT and similar problems.

### Weaknesses
* It's unclear if the method works for non English directions. From what I could understand one could load more than one language module and translate across language families. Could you clarify that?
* All the evaluation relies on reference-based automated metrics. There's no human evaluation to validate if the findings automated metrics hold

### Questions
Questions made above

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
5

---

## Human Reviewer 2

### Summary
The paper presents plug-and-play adapter modules to improve language scalability for multilingual machine translation with LLMs. Unlike existing models that massively scale the support for new LPs without delivering high quality on all supported languages, the authors claim that using language-specific (LS) modules (for language groups) can reduce interference between languages during training and improve learning and downstream performance. Only the base model and the corresponding language-specific module are activated for a specific LP during inference. 

The authors train the complete model (base +LS modules) in three pre-training and two post-training stages. For the last post-training stage, they also propose adaptive rejection to reduce over-rejection, a phenomenon that happens when the preferred and dispreferred translations have high overlap.

Results on FLORES shows that their model achieves consistently high-quality translations across language subgroups as compared to  baselines.

### Strengths
1. The authors propose plug-and-play adapters to improve language scalability for LLM-based multilingual machine translation and present a recipe for training such models. 
2. The multilingual preference dataset (to be released) can be a valuable resource to the community.
3. Results on the FLORES benchmark show that the proposed approach results in superior quality than baselines across language subgroups, showing the efficacy of their approach.

### Weaknesses
1. Using language-specific adapters for multilingual MT is not new.  The paper doesn't establish much connection to prior work that has been done before the LLM era to improve multilingual machine translation by incorporating various modular and adaptable approaches. This is important even if the methodologies and technical setups differ. 

2. The ARPO loss introduced in Section 4  doesn't address the core issue which is the granularity at which the preference loss is defined and applied. Instead of diluting the signal from dispreferred response with a penalty when the preferred and dispreferred are close, a more meaningful learning signal would be to help the model focus on differences between the two texts at a finer granularity.  This can be especially important where preferences hinge on subtle but critical nuances of translation. For now, ARPO risks being a temporary fix.

### Questions
1. L249-250 Why is parallel text referred to as a pseudo-monolingual dataset?

2. L213-215: what does accurate classification mean here?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces X-ALMA, a multilingual machine translation model that aims to achieve top-tier performance across 50 diverse languages, regardless of their resource levels. It is friendly to mid- and low-resource languages. The model uses a plug-and-play language-specific module architecture to prevent language conflicts during training and a carefully designed training regimen with novel optimization methods to maximize translation performance. Experimental results demonstrate that X-ALMA outperforms existing multilingual LLMs in every single translation direction on the FLORES-200 and WMT’23 test datasets according to COMET-22. The paper's contributions include the design of X-ALMA's architecture and training methodology.

### Strengths
The paper introduces a new approach to multilingual machine translation, featuring a plug-and-play module architecture that effectively addresses language conflicts during training. Additionally, the ARPO method represents a creative combination of existing ideas, aiming to optimize translation performance.
Extensive experimental evaluations demonstrate the effectiveness of the X-ALMA model, achieving superior performance compared to existing open-source multilingual LLMs on multiple language pairs. The paper provides clear descriptions of the background, related work, experimental setup, and analysis of results, making it easy to understand and follow.
This work has the potential to make a significant impact on multilingual machine translation, particularly in terms of delivering high-quality translations for resource-limited languages. 
The paper is well-structured and clearly written, providing readers with a comprehensive understanding of the research work.

### Weaknesses
The paper presents a comparison with the state-of-the-art in LLM-based machine translation. I am curious about how this compares to the state-of-the-art results of encoder-decoder NMT models that are trained on specific languages.

### Questions
On page 5, in Pre-Training Stage 2: For the 10B used in fine-tuning, are an equal number of tokens taken from each language in the group, or are they taken proportionally?

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4