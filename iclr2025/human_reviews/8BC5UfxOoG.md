## Human Reviewer 1

### Summary
This paper examines how example selection in in-context learning (ICL) can amplify biases in Large Language Models (LLMs). The authors make three key discoveries: example selection with high accuracy doesn't guarantee low bias, ICL example selection amplifies existing LLM biases, and it contributes to spurious correlations. To study these effects, they create EEC-paraphrase, a new sentiment classification dataset, and propose ReBE (Remind with Bias-aware Embedding), a novel debiasing method that combines contrastive learning with prompt tuning. Their results show that ReBE effectively reduces bias without significantly compromising accuracy while remaining compatible with existing example selection methods.

### Strengths
1. The paper identifies and investigates a new problem: how example selection in ICL affects model bias, an important angle that previous work has overlooked.

2. They creates a new dataset (EEC-paraphrase) that improves upon existing bias evaluation datasets by using more natural language and also proposes an innovative solution (ReBE) that creatively combines contrastive learning with prompt tuning.

3. . The authors test their findings across 8 different LLMs of varying sizes and 4 distinct example selection methods, using multiple bias metrics (AvgGF, MaxTG, MaxFG) to ensure comprehensive assessment. The authors also provide solid theoretical grounding to explain why their approach is effective.

### Weaknesses
1. A major methodological weakness lies in the dataset construction. Using GPT-3.5-Turbo to create EEC-paraphrase raises questions about potential inherited biases and quality control. The paper lacks discussion of human evaluation or validation of the paraphrased outputs.

2. The experimental evaluation would benefit from broader comparisons. The absence of comparisons with existing debiasing methods and fine-tuning approaches makes it difficult to fully assess ReBE's advantages. Including baselines like data augmentation or adversarial debiasing would provide valuable context.

### Questions
1. I'm concerned about potential biases introduced by using GPT-3.5-Turbo to create EEC-paraphrase. Could you describe your validation process and any controls implemented to ensure dataset quality? Human evaluation results would be particularly valuable.

2. The paper's findings about bias amplification are interesting but limited to sentiment classification. Have you explored whether these patterns hold true for other tasks like toxicity detection or text classification? Even preliminary results would help assess generalizability.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
5

### Confidence
2

---

## Human Reviewer 2

### Summary
This paper investigates how example selection methods for in-context learning (ICL) affect the biases of large language models (LLMs). The authors construct a new sentiment classification dataset, EEC-paraphrase, and discover three key findings: 1) high accuracy in example selection does not guarantee low bias, 2) example selection amplifies existing LLM biases, and 3) example selection contributes to spurious correlations in LLMs. To address these issues, they propose ReBE (Remind with Bias-aware Embedding), a method that uses contrastive learning and prompt tuning to mitigate biases while maintaining accuracy. The authors conduct extensive experiments using eight different LLMs and four example selection baselines. Their proposed ReBE method appears to reduce maximum bias values without compromising accuracy and demonstrates compatibility with existing example selection approaches.

### Strengths
- The paper raises an important and previously unexplored concern about how example selection might affect model bias.
- The experiments are well-structured across multiple models and methods, making the results more reliable.
- If the findings hold true, they have important implications for how ICL should be used in practice.

### Weaknesses
- The authors claim that their EEC-paraphrase dataset consists of sentences that are more complex and natural. However, there is no validation for this claim. The instructions to ChatGPT was to "expand the following sentence in more complex words." I'm skeptical that leads to sentences that are "more complex" and "natural". Furthermore, using an LLM to create sentences that are then used to study biases in other LLMs is not the right way to design this experiment.
- No ablation studies showing which components of ReBE are actually responsible for bias reduction. The paper claims ReBE "doesn't significantly compromise accuracy" but doesn't define what constitutes significant compromise. The authors haven't shared any statistical significance tests in the paper.
- The paper is not well written and the exposition needs a lot of improvement. Every figure and table should be clearly explained in text.
- All experiments are on a single task type (sentiment analysis). The authors haven't done any comparison to simpler bias mitigation approaches when many exist.

### Questions
- I'd like you to do some dataset validation, such as through human evaluation of the paraphrased sentences to ensure quality and meaning preservation, comparison with real-world text samples to validate ecological validity, and analysis of potential artifacts introduced by GPT-3.5 paraphrasing.
- I'd also like to see more methodological validation. Please include ablation studies isolating each component of ReBE. Share statistical significance testing for all reported results. I'd prefer to see some comparison with simpler debiasing approaches as well.
- Especially since you're using GPT-3.5 paraphrasing and not human annotation, could you come up with similar datasets for different types of tasks beyond sentiment analysis and conduct experiments there as well?
- There are many issues with the presentation. There is little explanation for figures or tables on how to read them. Then the writing itself has issues. Just as an example, why is Albaquerque et al the citation for Spurious correlations on Line 95? Then that is followed by, "Typical spurious correlations include stereotypes such as "He is a doctor; she is a nurse." That's not great academic writing. Please go over the paper with a red pen.

### Soundness
2

### Presentation
1

### Contribution
2

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
The paper introduces Remind with Bias-aware Embedding (ReBE), a method aimed at mitigating bias in LLMs by addressing spurious correlations through prompt tuning. Using a newly constructed version of Equity Evaluation Corpus, the authors evaluate how in-context learning (ICL) example selection influence biases related to gender and race. This paper considers a variety of GPT, OPT, and Llama models and study bias amplification using several prompt selection techniques: Random example selection, Perplexity, Similarity and DPP. Results show that ICL prompt selection generally does not increase average bias regardless of the model/example selection method, as measured by Average Group Fairness, it can amplify maximum bias levels measured by Maximum TPR Gap and Maximum FPR Gap.  In order to reduce the increase in maximum bias levels caused by ICL example selection, the authors introduce ReBE. ReBE is designed using a contrastive loss function that encourages bias-aware embeddings, aiming to reduce biases without significantly impacting model accuracy.

### Strengths
1. Novelty in Addressing Bias in ICL Example Selection: The paper tackles an underexplored problem, focusing on how example selection of ICL prompts with example selection amplify bias in LLMs. Demonstrating that different example selection methods increase maximum bias is an important finding. Further, disentangling native bias in the parameters of the model with ICL example selection bias provides a more holistic evaluation of bias in LLM outputs.

2. Effectiveness of ReBE: Figure 7 demonstrates that ReBE reduces bias in LLMs more than several other example selection methods as measured by Max FG. Further, the accuracy of using DPP + ReBE does not appear to impact the accuracy of the LLM significantly.  

3. Comprehensive Experimental Setup: The authors conduct experiments across multiple model types and sizes (e.g., LLaMA-2, OPT), which strengthens the generalizability of the findings.

### Weaknesses
1. The relationship between ICL example selection and bias amplification is complex and not as straightforward as the authors present it. Figure 2 demonstrates that example selection reduces average bias across most models. The claim that “all LLMs exhibit an increase in the maximum gender or race bias value with random-based example selection for ICL” (lines 217-218). This statement may be too bold of a claim based on their mixed results of bias amplification.

2. $\textbf{ReBE does not always significantly reduce bias}$: While ReBE reduces bias in many cases, its improvements over DPP are limited. Figure 7 shows that the decrease in bias is modest, which may raise questions about ReBE’s overall efficacy. 

3. There is no Analysis on the impact of increasing the number of ICL examples. Conducting bias analysis for ICL example selection at a different number of shots for all ICL example selection methods in this paper would be an important addition to the paper. Some analysis of the impact of the number of shots is conducted in Figure 8, but this is limited only to ReBE.

### Questions
1. Although ReBE is presented in the paper, Figure 6 is a little confusing and is not fully explained either in the main body of the paper or the Figure caption. 

2. How does increasing the number of ICL examples impact bias in other example selection methods? 

3. Would the authors be willing to elaborate on the additional train time required to add ReBE to existing methods?

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