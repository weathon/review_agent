# How to train data-efficient LLMs

- Avg Score: 6.80
- Decision: Accept (Poster)
- Scores: 8, 6, 8, 6, 6

## Abstract
The training of large language models (LLMs) is expensive. In this paper, we study data-efficient approaches for pre-training LLMs, \ie, techniques that aim to optimize the Pareto frontier of model quality and training resource/data consumption. We seek to understand the tradeoffs associated with data selection routines based on (i) expensive-to-compute data-quality estimates, and (ii) maximization of coverage and diversity-based measures in the feature space. Our first technique, AskLLM, leverages the zero-shot reasoning capabilities of instruction-tuned LLMs to directly assess the quality of a training example. To target coverage, we propose density sampling, which models the data distribution to select a diverse sample. Testing the effect of $22$ different data curation techniques on the pre-training of T5-style of models, involving hundreds of pre-training runs and post fine-tuning evaluation tasks, we find that AskLLM and density are the best methods in their respective categories. While coverage sampling techniques often recover the performance of training on the entire dataset, training on data curated via AskLLM consistently outperforms full-data training---even when we sample only $10$\% of the original dataset, while converging up to $70$\% faster.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper explores how to balance data coverage and model quality in large language model training. The goal is to find sampling strategies that minimize the amount of data needed while still maximizing model performance, essentially pushing the Pareto frontier of efficiency.

### Strengths
1.The study is timely and relevant, offering solid empirical evidence that directly contributes to the ongoing conversation on efficient LLM training.
    
2.The experiments are extensive — covering a wide range of model sizes (from 60M to 800M parameters) and 22 different sampling strategies — giving the conclusions strong credibility.

### Weaknesses
1.All experiments are conducted on T5-style encoder–decoder architectures, with no validation on the more widely used decoder-only models (like GPT). This limits the generalizability of the results.
    
2,Although ASK-LLM demonstrates efficiency, it still requires scoring every single sample in the dataset, which makes it computationally expensive. The practical scalability and real-world deployment cost are not fully addressed.

### Questions
1.The ASK-LLM approach is innovative — using the model itself to assess data quality rather than relying on external heuristics — but it may risk creating a “self-reinforcing bubble,” where the model’s own biases limit data diversity.
    
2.The ASK-LLM method is computationally costly. It might be worth exploring hybrid strategies, such as integrating active learning to selectively annotate boundary samples and iterate.
    
3.While the empirical evidence is rich, the paper would benefit from a clearer theoretical explanation of why ASK-LLM captures data quality better than traditional perplexity-based filtering.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores how to efficiently train Large Language Models (LLMs), with its core focus on investigating the impact of data selection on pre-training efficiency and model quality. It proposes two new data sampling methods:  

#### ASK-LLM (Quality-Based)  
This approach that leverages the zero-shot reasoning capability of an existing, instruction-tuned LLM (a "proxy model"). Through specific prompts, it "asks" this proxy model to directly evaluate whether each training sample is "information-rich and high-quality."  

#### DENSITY (Coverage-Based)  
This method aims to maximize data diversity. It models the data distribution and prioritizes selecting samples from regions that are "under-represented" in the distribution.  


The paper conducts large-scale empirical research: it tests up to 22 different data curation techniques on the T5 model, involving hundreds of pre-training runs and thousands of fine-tuning evaluations.  

It draws key conclusions regarding "quality" vs. "coverage":  
- Coverage-based sampling (e.g., DENSITY) can usually "recover" or "match" the performance achieved by training with the full dataset.  
- Quality-based sampling (e.g., ASK-LLM) is capable of "surpassing" the performance of training with the full dataset.  

Experiments show that even when using only 10% of the original data, ASK-LLM trains models that outperform those trained on 100% of the data—and with a 70% faster convergence rate.

### Strengths
The greatest strength of this paper lies in its proposal of the innovative ASK-LLM method.  

Instead of adhering to traditional perplexity or classifier scoring, it pioneers the use of the zero-shot reasoning capability of an existing LLM. Through specific prompts, it "asks" the model to directly determine whether a data point contains "informational signals."  
ASK-LLM explicitly resolves several key flaws of traditional perplexity filtering:  
- **Avoiding meaningless content**: Perplexity filtering tends to select high-frequency, repetitive "nonsense" (e.g., "Example 14" and "15"), while ASK-LLM can identify such content as low-quality.  
- **Preserving context**: Perplexity filtering incorrectly selects context-deficient fragments like "questions without answers," whereas ASK-LLM can properly recognize these as non-"informational" data.  
- **Retaining "long-tail knowledge"**: Perplexity filtering mistakenly discards valuable yet "niche" knowledge containing rare words (e.g., "Example 17"), but ASK-LLM can leverage its reasoning ability to identify and retain such "long-tail knowledge."  

The paper features an impressive volume of experiments, making its conclusions highly credible.  
The paper compares up to 22 different data curation techniques.  To complete this comparison, the researchers conducted 220 independent pre-training runs and 1,100 downstream task fine-tuning evaluations. In the field of LLM pre-training, such a large-scale comparison is extremely rare and valuable.  
Experiments were conducted on two model sizes (T5-Small and T5-Large) and five data sampling rates (10% to 80%), ensuring the robustness of the conclusions.  

The paper clearly answers the core question raised in the introduction: "Which matters more, Quality or Coverage?"  
- The upper limit of coverage-based sampling (e.g., DENSITY) is to **"recover" or "match"** the performance of full-data training.  
- Quality-based sampling (e.g., ASK-LLM) can **"surpass"** the performance of full-data training.  

The paper does not stop at superficial results but provides in-depth analyses:  
- **Correlation Analysis (Figure 7)**: By calculating the correlation of scores from different samplers, the paper proves that ASK-LLM, Perplexity, and DENSITY indeed select distinctly different data—confirming that ASK-LLM represents a brand-new evaluation dimension.  
- **Qualitative Analysis (Appendix H)**: It provides numerous real cases of data selected or discarded by ASK-LLM, intuitively demonstrating why it outperforms perplexity filtering.  
- **Scoring Model Scaling (Figure 6)**: The paper finds that using a more powerful LLM (from Small to XXL) as the "judge" for ASK-LLM leads to better data filtering results. This indicates that the ASK-LLM method itself has excellent scalability and can become more effective in the future as "judge" models iterate.

### Weaknesses
1. Concerns Regarding Model/Data Scale and Method Generalizability (Scaling & Generalizability)  

Model Scale  
T5-Large (800M parameters) is indeed hardly qualified to be called a "Large Language Model" (LLM) today—even in the 2023-2024 period. Core challenges and phenomena in pre-training (such as emergent abilities) are most pronounced primarily in models with tens of billions or even hundreds of billions of parameters. Therefore, it is questionable whether the "optimal" data strategy derived from an 800M-parameter model can be directly generalized to models with 70B or 1T parameters.  

Data Quality and Generalizability  
C4 is a relatively "noisy" corpus that has not undergone fine-grained filtering. For such datasets, the room for improvement through "data quality filtering" is inherently large (a scenario akin to "picking the best from a bad lot").  

This gives rise to a risk: the remarkable success of ASK-LLM—reducing data volume by 90%, cutting training time by 70%, while achieving better performance—may overestimate the method’s marginal benefits when applied to cleaner, higher-quality datasets. Examples of such datasets include RefinedWeb, Dolma, or the "textbook-level" data used for Phi-2. If a dataset is already of high quality, how much additional improvement can ASK-LLM bring? This is a question the paper cannot answer.  

2. Another issue is that the related works cited in the paper are relatively outdated, with a lack of recent relevant studies. This makes it difficult to assess whether the paper still holds innovativeness.

### Questions
1. What is the true cost-benefit trade-off?  
In Section 6 and Appendix C.1, the paper brilliantly argues that the high scoring cost of ASK-LLM is a "one-time amortizable" investment.  

My question is: Given today’s dataset scales of 10T+ tokens, using a powerful proxy model (7B+ parameters) to perform a full inference on every training sample makes this "one-time" cost extraordinarily high.  

The paper claims that training speed is accelerated by 70%, but if the scoring process itself requires computational resources equivalent to (or even exceeding) the total training time, what would the net benefit of this method be in practice?  

I would very much like to see a more direct cost analysis: (scoring cost + 70% of training cost) vs. (100% of training cost). The paper fails to provide this critical comparison.  


2. What exactly is the "Gemma 3 variant"?  
The paper’s strongest (and most clever) response to concerns about "scalability" comes in Section 6, where it states: "...a variant of our ASK-LLM technique was used in the development of the Gemma 3 family of models."  

My question is: The paper’s key scalability claim relies entirely on the word "variant."  

How different is this "variant" from the method proposed in the paper (Figure 3)?  

Does it still involve "asking" with a complex prompt containing multiple criteria (informativeness, format, safety)? Or has it been reduced to a small component within a larger, more complex filtering pipeline?  

If this "variant" is significantly different from the original method, the generalizability of the paper’s main conclusions (drawn from T5-800M) to the scale of Gemma 3 becomes far less certain.  


3. Prompt Sensitivity  
The entire magic of ASK-LLM is encapsulated in the prompt shown in Figure 3. This prompt itself is a complex, multi-dimensional instruction that requires the model to simultaneously judge "informational signals," "format," "knowledge," and "harmful content."  

My question is: How sensitive are the results to the wording of this prompt?  

What would happen if I removed the criterion "contain usable knowledge" and only kept "well-formatted"?  

If I only asked "whether it contains harmful content," would it function as a safety filter?  

The paper provides no ablation study on this newly introduced, complex "prompt hyperparameter."  


In summary, these questions do not negate the merits of the paper, but they are the answers I would most want to know when considering deploying this method in practical production or conducting follow-up research.

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
3

### Summary
This paper studies data-efficient pretraining of large language models (LLMs) by systematically evaluating two data selection strategies integrated into the pretraining pipeline.The proposed ASK-LLM method uses an instruction-tuned model to score each pretraining sample via prompting for quality judgment, selecting data with high predicted informativeness and linguistic coherence. In contrast, DENSITY estimates sample representativeness using embedding-space density and selects a diverse subset that maximizes coverage of latent topics. Both methods are applied before model training as a data filtering stage, followed by standard T5 pretraining on the selected subsets under a fixed compute budget. The study compares 22 sampling schemes on subsets of the C4 corpus using T5-small and T5-large, and evaluates downstream on 111 tasks. Results show that ASK-LLM substantially improves data efficiency—matching or surpassing full-data training with 10–60% of data and converging up to 70% faster—while DENSITY performs best among coverage-based baselines. Random sampling remains competitive but inferior to the LLM-guided quality filtering approach.

### Strengths
- The paper studies how different data sampling strategies influence the efficiency of LLM pretraining under a clearly defined and reproducible setup.
- It conducts extensive and controlled experiments, covering 22 sampling strategies, over 200 pretraining runs, and 1,100 downstream fine-tunings under a fixed-compute regime.
- The methodology is clearly framed through a contrast between quality-driven (ASK-LLM) and coverage-driven (DENSITY) sampling, offering a structured perspective on data selection principles.
- Empirical results are consistent across model scales and sampling ratios: ASK-LLM yields larger gains in low-data regimes, while DENSITY maintains stable coverage performance, supported by convergence and ablation analyses.
- The experiments include explicit reporting of compute budgets, dataset proportions, and evaluation metrics, allowing clear interpretation and comparison of the data-efficiency results.

### Weaknesses
- The ASK-LLM sampler requires one inference per training example, which introduces notable computational cost. While the authors mention this can be amortized across multiple runs, the initial scoring overhead may affect scalability to larger datasets.
- The baselines do not include comparisons with more advanced recent methods such as Harnessing Diversity for Important Data Selection in Pretraining Large Language Models (ICLR 2025).
- All experiments are conducted on encoder–decoder (T5) models using text-only corpora, and the generalization of the findings to decoder-only or multimodal architectures has not been evaluated.

### Questions
- Did the authors experiment with alternative prompt formulations or response schemes for ASK-LLM—for example, using scalar or graded scoring instead of binary yes/no judgments? If not, how confident are they that the binary formulation captures sufficient quality signal?
- How sensitive are the ASK-LLM filtering results to the choice of the LLM judge? The paper suggests that different Flan-T5 judge sizes yield similar filtering behavior—could the authors clarify where this consistency arises from?

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
This paper investigates data-efficient pre-training methods for LLMs to reduce high training costs. It contrasts two primary data selection strategies: data quality and data coverage. The authors introduce two novel methods to represent these strategies:
1. ASK-LLM: A quality-based sampler that uses an instruction-tuned LLM to perform a zero-shot, prompt-based assessment of a data point's quality.
2. DENSITY: A coverage-based sampler that selects a diverse set of examples by modeling the data distribution in the feature space.

In a large-scale experiment testing 22 different curation techniques on T5-style models, ASK-LLM was the top performer. The key finding is that while coverage-based samplers typically match the performance of training on the full dataset, the quality-based ASK-LLM consistently outperforms full-data training.

### Strengths
1. The paper's conclusions are supported by a massive-scale experiment. The authors tested 22 different data curation techniques , involving 220 pre-training runs and 1,100 distinct fine-tuning runs , all evaluated on 111 downstream tasks. This comprehensive and costly analysis provides high-confidence evidence for their claims.
2. The primary strength is the clear finding that quality-based filtering (ASK-LLM) can train models that outperform models trained on the entire dataset, even when using as little as 10% of the data. This is a significant result because it shows data curation is not just about matching full-data performance more efficiently, but about achieving a new, higher performance ceiling.
3. The proposed ASK-LLM sampler is a novel and intuitive method that leverages an LLM's reasoning to score data quality. This approach is shown to be superior to traditional perplexity filtering because it captures a different, more valuable signal. The paper demonstrates that ASK-LLM scores have "almost no ranking correlation" with perplexity scores and that it correctly identifies high-quality, niche-topic examples that perplexity filtering wrongly penalizes.

### Weaknesses
1. The ASK-LLM method requires one full LLM inference pass for every single example in the training corpus. For a dataset like C4 (364M examples), this is a massive computational cost that could rival the cost of training itself. While the authors argue this cost is "amortized" over many training runs, this presents a very high, practical barrier to entry.

### Questions
1. The paper's evaluation relies entirely on post-finetuning performance across 111 tasks . How do you expect the ASK-LLM curation to affect the zero-shot and few-shot reasoning capabilities of the pre-trained models? Is there a risk that filtering for "informative" data (as defined by the prompt) optimizes for finetuning performance at the expense of the broad, general-purpose in-context learning abilities that are a key benefit of pre-training? 
2. The ASK-LLM sampler requires a full inference pass over the entire dataset, which is a significant, up-front computational cost. While you argue this cost is amortized, could you provide a more concrete cost-benefit analysis? For example, using the T5-Large model, how many compute-hours did the ASK-LLM (XL) scoring pass on the full C4 dataset take?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
- This paper addresses the critical challenge of efficient data selection in LLM pre-training. The core research question investigates the trade-offs between data selection strategies based on data quality versus those based on coverage. 
- To explore this, the authors propose two novel data curation techniques: (1) ASK-LLM, a quality-based sampler that leverages the zero-shot reasoning of an instruction-tuned LLM to directly score the informativeness of each training example, and (2) DENSITY, a coverage-based sampler that uses kernel density estimation on data embeddings to select a diverse subset of the data.
- The authors conduct an extensive empirical study by pre-training T5-style models (60M and 800M parameters) on various subsets of the C4 dataset curated by 22 different methods.

### Strengths
- The motivation of this paper is clear: selecting high-quality and high-coverage data for efficient pretraining.
- While model-based data selection has been explored, the use of an instruction-tuned LLM's explicit, prompt-based reasoning to assess the quality of individual pre-training samples is a novel and powerful concept.
- The experimental results are highly insightful, demonstrating that (1) compared to other model-based methods (e.g., PPL), the proposed ASK-LLM leverages the reasoning and context ability from the instruct-tuned LLM for long-tail high-quality data selection. (2) With the improvements of the scoring model, there are increasingly beneficial effects as the scaling of the to-be-trained model. (Perplexity filters do not seem to exhibit such trends.) 
- The paper is well-written, well-organized, and easy to follow.

### Weaknesses
- This work conducts all experiments on the T5 series encoder-decoder LLMs. As of late 2025, the vast majority of state-of-the-art open-source research and applications are centered on decoder-only LLMs, like Qwen/Llama/Gemma/Mistral. While the authors acknowledge this limitation, the absence of any experiments on a decoder-only architecture makes it difficult to assess the direct generalizability of the findings. 
- The paper argues that the high inference cost of ASK-LLM is amortized over multiple training runs. However, scoring a trillion-token dataset is much expensive. The paper would benefit from a more concrete cost analysis. For example, a simple table estimating the TPU/GPU-hours required to score the full C4 dataset with Flan-T5-XL versus the compute saved from the accelerated T5-Large training run would provide a much clearer picture of the return on investment.

### Questions
Same as the weaknesses.

### Soundness
4

### Presentation
4

### Contribution
3
