# Interpretable Embeddings with Sparse Autoencoders: A Data Analysis Toolkit

- Decision: Reject
- Scores: 4, 8, 2, 6

## Abstract
Analyzing large-scale text corpora is a core challenge in machine learning, crucial for tasks like identifying undesirable model behaviors or biases in training data. Current methods often rely on costly LLM-based techniques (e.g. annotating dataset differences) or dense embedding models (e.g. for clustering), which lack control over the properties of interest. We propose using sparse autoencoders (SAEs) to create $\textit{SAE embeddings}$: representations whose dimensions map to interpretable concepts. Through four data analysis tasks, we show that SAE embeddings can find novel data insights while offering the controllability that dense embeddings lack and costing less than LLMs. By computing statistical metrics over our embeddings, we can uncover insights such as (1) semantic differences between datasets and (2) unexpected concept correlations in documents. For example, by comparing model responses, we find that Grok-4 clarifies ambiguities more often than nine other frontier models. Relative to LLMs, SAE embeddings uncover bigger differences at 2-8× lower cost and identify biases more reliably. Additionally, SAE embeddings are controllable: by filtering concepts, we can (3) cluster documents along axes of interest and (4) outperform dense embeddings on property-based retrieval. Using SAE embeddings, we study model behavior with two case studies: investigating how OpenAI model behavior has changed over new releases and finding a learned spurious correlation from Tulu-3's (Lambert et. al) training data. These results position SAEs as a versatile tool for unstructured data analysis and highlight the neglected importance of interpreting models through their $\textit{data}$.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
Large-scale text corpora analysis is challenging. This paper proposes using Sparse AutoEncoders (SAEs) to create interpretable embeddings for analyzing large corpora. 
Extensive quantitative experiments suggest that SAE embeddings are useful and versatile tools for data analysis, benefiting multiple applications, including dataset diffing, correlation, clustering, and retrieval.

### Strengths
- Some analyses are interesting, especially the LLM characteristics change over different generations of GPT (discussed in Section 5.1), and the identification of spurious correlation in Tulu-3’s SFT dataset (in Section 5.2). This evidence also demonstrates the usefulness of the proposed method.
- Compared to LLM-based inspection, SAEs are more economical, particularly for data diffing tasks.
- Experiments are detailed and (apparently) easy to reproduce, improving credibility and reusability.

### Weaknesses
- Organization and readability can be improved; heavy cross-referencing to tables/figures disrupts flow. A streamlined structure would significantly help comprehension.
- The procedure for producing interpretable activation vectors is not very clear. I am confused about how to define the meanings for each feature. For example, in Fig. 1, why does “feature 1” correspond to nouns and “feature 2” to animals?

### Questions
- How are 61,521 latent descriptions obtained from a dictionary of size 65,536? What is the exact pipeline for deriving concrete descriptions per feature?
- For lines 302–303 and Fig. 8 (retrieval), how are latent labels produced? Are they LLM-generated?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper utilizes pretrained SAEs' latens to build an embedding model that is able to capture property of interest and also allows interpretability whereas dense embedding models lack both, and are trained to capture semantics which does not allow full controllabilitiy. With their methodology, they are capable of doing dataset diffing more efficiently and accurately and it enables exploration of unknown correlation between different concepts that are often missed by LLMs. Moreover they adapt these embeddings to other downstream tasks such as clustering and retrieval. Lastly, they analyze two case studies which are the behavior change of openai models illustrated by their personification ability, and analysis of tulu3 sft data in which they find that math/latex like concepts correlate with the hoping and they are able to trigger this behaviour by using the correlated concepts. Overall, they modify the popular SAE framework-which is primarily used to investigate model behaviour- to generate embeddings that could be used in various NLP tasks while often outperforming or being on par with LLMs or dense embedding models.

### Strengths
* Paper is well written and easy to follow, figures are creative and helpful.
* Even though SAEs are well known in mech interp, adaptation of them as embedding models is both interesting and novel.
* Experimental setups are clearly explained and diverse, and claims are coherent with the findings.

### Weaknesses
Major
* Lack of ablations on SAEs(size,corpora etc), and similarly for reader LLM, and also diversity of datasets.

Minor 
* A lot of the results are in the appendix, so there's a lot of back and forth while reading.

### Questions
1)How much does the SAE or reader LLM impact quality of embeddings, have you done any ablations on them?
2) The paper primarily uses chatstyle prompts, have you guys explored any other datasets or prompts that could be treated as somewhat out-of-distribution in dataset-diffing and correlation experiments?
3) It is known that malicious texts in the pretraining do poison LLMs, how can we adapt this framework to large scale corpora filtering?
4) As the paper mentions, SAE embeddings are mostly property-based. Could we improve current dense embeddings by training SAEs on them, or by taking a trained SAE from a model like Llama3-70B or Gemma (which already have SAEs) and converting it to dense embeddings with something like LLM2Vec? Have you seen any cases where this actually helps, or are they just naturally different approaches?
5) Feature relabelling is critical, what if we dont know the distribution of the data so we cant recorrect them which is plausible in the deployment, how much will the quality of embeddings degrade?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores the use of Sparse Autoencoders (SAEs) trained on LLM hidden states to create interpretable embeddings for text data. Each dimension in the SAE embedding is supposed to represent a human-interpretable feature, which the authors use for four text analysis tasks: comparing datasets, finding correlations, clustering, and property-based retrieval. The paper presents two important case studies: The first tracks behavioral changes across ChatGPT versions, showing increasing nuance and self-critique in responses, more personalized follow-ups, and descriptions that personify objects more frequently. The second analyzes Tulu-3’s fine-tuning data and finds that certain prompt formats, such as numbered lists, LaTeX math, and role-playing instructions, consistently cause it to end with the phrase “I hope it is correct.” The paper traces this to a specific training subset and confirm that these prompt patterns directly trigger the behavior in the Tulu-3 model.

### Strengths
The paper introduces a novel and creative application of SAEs beyond their typical role in LLM interpretability to the domain of textual data analysis. I think SAEs are a great choice as a data analysis toolkit for the following reasons:  the interpretable and sparse embeddings offer greater controllability compared to dense embeddings, like enabling pre-filtering of features for targeted analysis of specific properties. Further, SAEs can capture implicit features of chat dialogues beyond coarse semantic content, e.g., sycophancy, anthropomorphism, or the presence of reasoning chains, which dense embedding models typically miss. This makes them a better candidate for analysing model training data/responses

The real-world use cases presented in the paper are great illustrations of the method’s practicality. Additionally, this paper presents a promising direction for unstructured data analysis of model-related data and might encourage interesting future research.

### Weaknesses
Overall, the experiments lack rigor, and the work feels preliminary (details below). I see this paper as a good proof-of-concept, and in its current state, it is more suitable for a workshop or a blog post.

I have listed some weaknesses along with some suggestions below (loosely in order of priority). Many of them are related to the four data analysis tasks. Personally, I think these tasks could be removed altogether. The paper would be stronger if it focused more on the case studies instead. You could then consider organizing the findings into clearer categories, such as debugging fine-tuning data or understanding model behavior both within and across model families.

* Table 6 provides a qualitative sanity check but lacks a quantitative or statistically grounded evaluation. Adding a measurable metric along with a comparison to the LLM baseline would increase the reliability of the claim.
* The correlation discovery method in Section 4.2 lacks a quantitative evaluation of its signal-to-noise ratio. Quantifying the proportion of trivial or false-positive correlations would help in assessing the method’s practical utility and how much manual checking is required
* In Section 4.4, the SAE retrieval setup introduces higher computational overhead due to latent-query dense similarity matching. Exploring alternative strategies that are more efficient would make the method more practical for real-world deployment.
* In Figure 2, the effect of correlated or redundant latents on difference detection is not analyzed. SAEs tend to produce correlated latents, and one should account for their impact on frequency differences.
* In Section 4.3, clustering results are only reported for one algorithm. Demonstrating consistent results across multiple algorithms would increase the reliability of the claim.
* Results are reported for only one SAE model. Testing multiple SAE models would make the findings more reliable and better support the claim that SAEs are a good choice.
* Prompt variation effects across all experiments are not studied. Testing different prompt formulations for the LLM stages and baselines would clarify how sensitive the results are to prompt design.
* The analysis workflows depend heavily on another LLM to summarize or relabel latent features, suggesting that SAEs alone are not sufficient for the intended tasks. Designing experiments that evaluate the effectiveness of SAEs in isolation would provide a clearer understanding of their independent capabilities and limitations.
* Many of the key results (Table 6, Figure 12, Table 10, Table 11, Table 12, Figure 13) are placed in the appendix rather than integrated into the main text. Bringing these results forward would make the paper’s main findings more accessible. For instance, while reading the paper, I had to move back and forth multiple times just to understand the results of each experiment. In similar vein, the figure that explains the flow for each of the analysis tasks (Figure 8) is also in the appendix. Including this figure in the main text would improve readability.
* In section 3, the paper does not describe how “activating examples” for a latent are selected. Describing the process would help better understand the SAE pipeline.
* The claimed 2-8x cost improvements appear less compelling since it is demonstrated only for the dataset diffing experiment. In fact, it feels somewhat misleading, as the clustering experiment actually shows an increase in computational cost.

Further, each task involves a fairly complex workflow and consequently could introduce compounding errors at multiple stages.

### Questions
* Are the relabelling costs included in the SAE method in Table 2?
* Is the precision for all tasks affected when the SAE training distribution differs substantially from the distribution of the target datasets?

### Soundness
2

### Presentation
2

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
- The authors propose using sparse autoencoders (SAEs) to generate interpretable text embeddings. Each SAE feature corresponds to an embedding dimension created by max pooling that feature's activations over a document, yielding embeddings where each dimension has a natural semantic interpretation.
- The authors demonstrate four applications of SAE embeddings: (1) data diffing to identify differences between text corpora, (2) discovering unexpected correlations between features, (3) clustering, and (4) retrieval.
- For each application, the authors provide both toy setting with known ground truth and open-ended exploratory analyses. In toy settings, they compare SAE embedding performance to corresponding dense embedding or LLM-based baselines.
- The authors lastly present two case studies: analyzing changes in OpenAI model behavior over time using data diffing, and discovering unexpected feature correlations in Tulu-3's post-training dataset.

### Strengths
- The idea of using SAEs to generate interpretable text embeddings feels novel and well-motivated.
- The authors cover an wide breadth of applications - data diffing, correlation discovery, clustering, and retrieval.
- The experiments have great coverage, including both toy settings with ground truth targets and real-world exploratory analyses. The authors make a solid effort to incorporate baselines (dense embeddings and LLM-based methods) for comparison.
- The real-world case studies find some interesting behaviors in deployed models (Grok, OpenAI models over time, Tulu-3)
- The author's use a LLM judge to verify many of the hypothesis generated by the diffing and correlations based methods - although implementation details are unclear

### Weaknesses
- The paper's breadth makes it challenging to communicate each experiment with sufficient depth. The main text requires constant cross-referencing with the appendix, and key details are often unclear or left for the reader to infer—for example, the latent relabeling procedure, synthetic dataset construction in Section 4.2, what constitutes a "hypothesis," and how hypotheses are verified.
- Many results follow a pattern of generating hypotheses, verifying some subset, and presenting the verified ones. It would be helpful to understand the selection process better: what is the false positive rate? How many proposed hypotheses were actually verified? Providing this context would help assess whether the positive results shown are representative or cherry-picked.
- Some applications feel less compelling and distract from the stronger results. For instance, the correlation findings reflect obvious dataset structure (e.g., Stack Exchange containing both QA and code). Similarly, the clustering results aren't falsified or verified, and they don't demonstrate the key advantage of SAE embeddings—the ability to cluster on specific interpretable concepts—instead just showing that SAE and dense embeddings produce different clusters.

Overall: my main concerns are about presentation and prioritization rather than the method or experiments themselves. Substantial revision for clarity would improve my assessment of the paper.

### Questions
- Can you provide more detail about the false-positive rate of hypotheses generated by your method? For the OpenAI behavior analysis and Tulu-3 data analysis, were the presented results selected from a larger set of hypotheses that were first verified? Understanding the selection process would help assess how representative these findings are. Similarly, was the choice to present only Grok4 behavior diffing results because it had the most differences, or because the other diffed models lacked different behaviors?
- You currently use one SAE configuration on one model, which is reasonable for a proof of concept. Do you have any preliminary experiments or intuitions about how method performance might change with different SAE configurations—both in terms of dictionary size and the size/complexity of the underlying model?
- Regarding the token usage comparisons in Table 2, as I understand it, the SAE tokens appear to be from Llama-70B, while the LLM baseline tokens are from whatever frontier model is used to review/process the data (Gemini 2 Flash). This does not seem to be a fair comparison unless I am missing something.

### Soundness
3

### Presentation
1

### Contribution
4
