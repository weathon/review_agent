# Learning on the Job: Test-Time Curricula for Targeted Reinforcement Learning

- Decision: Reject
- Scores: 4, 4, 4, 6, 6

## Abstract
Humans are good at learning on the job: We learn how to solve the tasks we face as we go along. Can a model do the same?
We propose an agent that assembles a task-specific curriculum, called *test-time curriculum* (TTC-RL), and applies reinforcement learning to continue training the model for its target task.
The test-time curriculum avoids time-consuming human curation of datasets by automatically selecting the most task-relevant data from a large pool of available training data.
Our experiments demonstrate that reinforcement learning on a test-time curriculum consistently improves the model on its target tasks, across a variety of evaluations and models.
Notably, on challenging math and coding benchmarks, TTC-RL improves the pass@1 of `Qwen3-8B` by approximately 1.8x on AIME25 and 2.1x on CodeElo.
Moreover, we find that TTC-RL significantly raises the performance ceiling compared to the initial model, increasing pass@8 on AIME25 from 40% to 62% and on CodeElo from 28% to 43%.
Our findings show the potential of test-time curricula in extending the test-time scaling paradigm to continual *training* on thousands of task-relevant experiences during test-time.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a way to build task-specific curricula to improve fine-tuning of large models to specific tasks. The approach called TTC-RL is a relatively simple "reorganization" of tasks to be executed during the fine-tuning of the model, heavily based on a previously published approach (SIFT) to make the selection of the tasks.

### Strengths
- Approach is relatively simple to implement and widely applicable to pretty much any fine-tuning task
- Extensive experimental evaluation is presented showing that the approach is better than not using curriculum.

### Weaknesses
- There is a gap in the literature covered by the paper, especially in regards to the most related works on Curriculum for RL. The authors should start with the two main surveys in the area, as they mention in passage the original Curriculum paper for supervised learning, and then go straight to the LLM-specific methods:

Da Silva, Felipe Leno, and Anna Helena Reali Costa. "A survey on transfer learning for multiagent reinforcement learning systems." Journal of Artificial Intelligence Research 64 (2019): 645-703.

Narvekar, Sanmit, et al. "Curriculum learning for reinforcement learning domains: A framework and survey." Journal of Machine Learning Research 21.181 (2020): 1-50.

In special, there are quite a number of Curriculum approaches that explore "transferability" metrics similar to what you are trying to accomplish with SIFT, from the probably most related "transfer potential":

Silva, Felipe Leno Da, and Anna Helena Reali Costa. "Object-oriented curriculum generation for reinforcement learning." Proceedings of the 17th international conference on autonomous agents and multiagent systems. 2018.

to the more classical but still likely adaptable to modern research "learned transferability"

Sinapov, Jivko, et al. "Learning inter-task transferability in the absence of target task samples." Proceedings of the 2015 international conference on autonomous agents and multiagent systems. 2015.

- Related to the last point, the experimental evaluation shows only the proposed approach X not using curriculum, which at this point is no surprise that results in a better performance. The experimental evaluation should consider other classical autonomous curriculum definition methods to show it even makes sense to develop an approach specific for LLM to bridge the gap from simple adaptation of the older methods.

- The final main weakness of the paper is that a very small portion of the paper is dedicated to explain the actual proposed method. The method description (one of the most critical portions of the paper), is described exactly between lines 158-179, barely 20 lines!!!, many of those simply pointing to other papers on the literature (specifically GRPO and SIFT) instead of simply listing step by step all operations - I appreciate the openness in showing specifically the parts of the method inherited from the literature, however it would make it much easier for the reader to have Algorithm 1 listing all steps and the exact equations being computed during the method (especially important for GRPO because you omit the KL term and the comment related to this in line 177 can be easily missed or forgotten by the reader).

### Questions
No specific question

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Test-Time Curriculum Reinforcement Learning (TTC-RL), a framework that enables large language models (LLMs) to autonomously curate task-specific curricula during test-time and improve via reinforcement learning. The core idea is to leverage a diverse corpus of training tasks, selectively retrieving those most relevant to a target task using the SIFT algorithm, and fine-tuning the model via GRPO. Experiments across math (AIME, MATH500), coding (Codeforces, CodeElo), and scientific reasoning (GPQA-D) benchmarks demonstrate significant performance gains—e.g., improving Qwen3-8B’s pass@1 on AIME25 by ~1.8× and on Codeforces by ~2.4×. The method also raises performance ceilings (e.g., pass@64 on Codeforces from 45% to 72%) and outperforms models with extended in-context reasoning. The authors argue that TTC-RL offers a compute-efficient alternative to scaling context windows.

### Strengths
1. The idea of test-time curriculum learning bridges the gap between large-scale pre-training and fixed test-time inference. It extends the "test-time scaling" paradigm by enabling models to meta-learn from targeted experiences, which is underexplored in LLM literature.

2. The paper provides extensive evaluations across multiple models (Qwen, Llama) and domains, showing consistent improvements over baselines like general-purpose RL post-training. The gains on competition-level benchmarks (e.g., Codeforces) are particularly impressive.

3. TTC-RL avoids human-curated data and reduces reliance on expanding context windows, highlighting a compute-optimal Pareto frontier between model specialization and resource costs.

4. The inclusion of "latent improvement" metrics addresses concerns about format overfitting, and ablations (e.g., SIFT hyperparameters, curriculum sizing) validate design choices.

### Weaknesses
1. Computational Cost: Training on dynamically selected curricula during test-time may incur non-trivial overhead, which is not quantified in terms of wall-clock time or energy usage.

2. Generalization Concerns: Specialization to target tasks comes at the cost of degraded performance on unrelated tasks (Figure 4), raising questions about balanced multi-task adaptation.

### Questions
1. You show TTC-RL beats 30k-token "thinking" models, but could in-context learning with smarter retrieval (e.g., RAG) close this gap?

2. For deployment, how would TTC-RL handle dynamically evolving tasks (e.g., user-specific queries)? Is there a risk of overfitting to transient test distributions?

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
4

### Summary
This paper proposes to use an auto-curriculum to sequence tasks for RL post training on a corpus, so that performance is improved on a set of target tasks. In particular, the paper proposes to use a previous method, SIFT, alongside a corpus of verifiable tasks that they have curated. Results across math and coding show that sequencing tasks in a curriculum leads to large improvements in tasks compared to randomly sampling these tasks. These results are contextualized by looking at performance differences across tasks, and with smaller models.

### Strengths
- The empirical results suggest that curricula for post-training can lead to large improvements in performance.
- Most of the paper is well-written, but some areas remain unclear (see below)

### Weaknesses
- Lack of novelty: performance can be attributed to a previously proposed method that is applied to generating curricula on the verifiable-corpus.
- There is comparatively little discussion of the proposed method. I also found Section 5 much less clear than the rest of the paper.
- No other auto-curriculum baseline: despite comparing the proposed curriculum method with post-training + random sampling, no other methods for constructing curricula are compared.

### Questions
- Figure 1: "While avoiding human-driven data curation" is not entirely a fair comparison, given that the curricula involves a corpus of data curated by humans
- Section 3.1: The main method proposed here is barely explained, and seems to be a straightforward application of existing methods.
   From the appendix, it seems 
- Section 3.1: From the appendix, it seems like SIFT is similar to a leave-one out estimator. Wouldnt this be extremely costly to compute with an LLM?
- Section 3.2: by introducing a corpus along with the method, there is some possibility that this corpus is particularly suited to your method or to curricula in particular. For example, it may be the case that the curriculum is effective because uniform random sampling (as in your post-training baseline) often samples a completely irrelevant task. However, this difference would disappear if both rl post training and the curriculum is allowed to learn from the entire corpus. Of course, this would be computationally challenging to investigate.
- line 313: Can you clarify what you mean by "improvement to the RL training algorithm"? Would these improvements also be used in the rl post-training baseline? If so, then I think this discredits the referenced claim that RL training is simply distilling pass@k, and doesn't fully explain the improvements brought by the curriculum.
- line 348: Is the cost of TTC-RL linear? Isnt there a cost associated with using SIFT?
- Section 4.2: While it is good to report this finding, I think this provides some evidence that there could be potential overfitting occurring by sequencing a narrow set of tasks. One thing that doesnt seem reported is whether the model actually degrades on tasks that are very different from the target task. (Figure 4, right seems to report "normalized accuracy" but not "accuracy change")
- line 399: I don't understand the lowerbound, P(correct) > P(accurate). Wouldnt correctness require accuracy as a pre-condition? This would imply that P(accurte) > P(correct). Also, what is the difference between accurate and well-formed: they are different in eq 1 but talked about as if theyre the same on line 398.

### Soundness
3

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
This paper proposes Test-Time Curriculum Reinforcement Learning (TTC-RL), a method for specializing large language models (LLMs) to target tasks at test time. The model automatically constructs a self-curated curriculum from a large training corpus (via SIFT embeddings) and fine-tunes itself using on-policy RL (GRPO).

### Strengths
- The idea of the paper, automatic self-curated test-time curricula (TTC-RL) for LLM specialization via RL, is novel.
- Strong empirical gains: Significant improvements across math and reasoning benchmarks done sample-efficiently.
- The Latent Improvement (LI) to measure real reasoning gains, not just formatting is intuitive and nicely thought.
- Reproducibility: Public verifiable-corpus (~280k tasks) and clear training pipeline.

### Weaknesses
- Primarily focuses on non-thinking models (Qwen3-8B, Llama-3.1-8B).
- TTC-RL relies on a fixed training corpus, limiting potential performance on completely novel target tasks.
- While more efficient than long-context scaling, RL-based test-time specialization still requires on-the-fly training, which could be impractical in some real-world applications.
- No code release: Makes it harder for the community to reproduce or extend the TTC-RL pipeline.

### Questions
- How does TTC-RL perform when target tasks are very different from training corpus tasks (e.g., out-of-distribution problems)?
- Can TTC-RL combine multiple target tasks with diverse difficulty levels effectively, or does it favor easier tasks?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a test-time curriculum method for RL-based fine-tuning of large language models. In contrast to manually curated datasets, TTC-RL selects relevant examples in a corpus that are closest to the test-time task based on their last-token embeddings, then fine-tunes the model using the selected samples. Experiments across a variety of benchmarks in math, coding, and scientific reasoning demonstrate that TTC-RL improves performance in pass@k, with k ranging from 1 to 64.

### Strengths
- The paper is very well-written, has an easy-to-follow structure, and includes figures that well describe the empirical advantages of the proposed framework.
- The proposed framework comes from an existing idea proposed for SFT, yet this work seems to be the first to apply this idea to test-time training via RL-based fine-tuning.
- Experiments cover a wide range of benchmarks, hence they provide comprehensive evidence that shows the advantages of TTC-RL in multiple base models. The results are well-documented as well.

### Weaknesses
- Although this is the first application of the nearest neighbor retrieval for sample selection in test-time training via RL, the idea is not novel at all.
- Although it makes sense to select samples from a corpus based on the similarity of last-token embeddings, the generated curriculum is fixed throughout test-time training. It does not depend on how the model performs at all, either.

### Questions
- Despite being knowledgeable about curriculum learning for RL and RLHF, I do not know much about test-time training. What is the practical purpose of such a framework, considering that at "test-time" there shouldn't be any training?
- The proposed approach selects samples based on embedding similarity, but this set is fixed during test-time training. Has it ever been considered to adaptively prioritize certain samples during TTC?

### Soundness
3

### Presentation
4

### Contribution
2
