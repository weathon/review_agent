# Automated Proof Generation for Rust Code via Self-Evolution

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
Ensuring correctness is crucial for code generation. Formal verification offers a
definitive assurance of correctness, but demands substantial human effort in proof
construction and hence raises a pressing need for automation. The primary obsta-
cle lies in the severe lack of data—there is much fewer proofs than code snippets
for Large Language Models (LLMs) to train upon. In this paper, we introduce
SAFE, a framework that overcomes the lack of human-written proofs to enable
automated proof generation of Rust code. SAFE establishes a self-evolving cycle
where data synthesis and fine-tuning collaborate to enhance the model capability,
leveraging the definitive power of a symbolic verifier in telling correct proofs from
incorrect ones. SAFE also re-purposes the large number of synthesized incorrect
proofs to train the self-debugging capability of the fine-tuned models, empowering
them to fix incorrect proofs based on the verifier’s feedback. SAFE demonstrates
superior efficiency and precision compared to GPT-4o. Through tens of thousands
of synthesized proofs and the self-debugging mechanism, we improve the capa-
bility of open-source models, initially unacquainted with formal verification, to
automatically write proofs for Rust code. This advancement leads to a signifi-
cant improvement in performance, achieving a 52.52% accuracy rate in a bench-
mark crafted by human experts, a significant leap over GPT-4o’s performance of
14.39%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper seeks to finetune a code-generating LLM to generate verification annotations for code.
Specifically, the authors target Verus, which is an SMT-backed automated-theorem-proving-style verifier for Rust code.
The key technical challenge that the authors thus need to overcome is that ths is a very low resource language,
so simple techniques such as finetuning aren't directly applicable; and even API-backed models do so poorly
on this task that naively distilling wouldn't help, either.
Instead, the authors basically bootstrap the finetuning process as follows:
- First, they generate a set of proof *specifications* for some Rust programs using GPT-4o. They then filter out specs which are "low quality", e.g. those which are always true.
- Then, they use these specifications to generate (against using GPT-4, with some expert-crafted task-specific prompts) proof *annotations* for a small subset of these specifications.
- Finally, they bootstrap a finetuning process from these initial annotations, training in each round the open-source model on the correct proofs it generated in the last round.
There's some additional bells and whistles, such as also training on incorrect proofs by framing it as auxiliary repair task, but I believe this summarizes the core idea.

In terms of the experiments, the authors share results both for a small, human-written benchmark and for GPT-transpiled versions of MBPP and SV.
At first glance I was a bit worried about the scale of this data, but given the novelty of the task and the relative lack of Rust datasets in the literature I actually commend the authors on their effort to collect as much data to evaluate on as possible.

### Strengths
I think the core ideas in this paper are very interesting, and that there are several good contributions here:
- Filtering the specifications based on a symbolic, deterministic score computed from the tests seems like the right thing to do, and I appreciate the brief ablation study of the impact of this step (section 4.2.4).
- The experiment in 466-474 provide further evidence for previous findings in the code generation literature about "sampling+greedy" self-debugging outperforming "greedy+sampling" (I recommend that the authors consider explicitly comparing these results to e.g. [1, 2]).
- Perhaps most importantly, verifying Rust code is not only potentially impactful but also (as far as I know) a completely novel task; kudos to the authors for going through the effort to collect all the data.


[1] 
Teaching Large Language Models to Self-Debug
Xinyun Chen, Maxwell Lin, Nathanael Schärli, Denny Zhou.
International Conference on Learning Representations (ICLR), 2024.

[2] 
Is Self-Repair a Silver Bullet for Code Generation?
Theo X. Olausson, Jeevana P. Inala, Chenglong Wang, Jianfeng Gao, Armando Solar-Lezama.
International Conference on Learning Representations (ICLR), 2024.

### Weaknesses
There are a few rather major flaws that make me hesitant to recommend the paper for acceptance in its current format.

One is that I find that the paper tries a little bit too hard to sell the novelty of this "SAFE" approach.
Bootstrapping finetuning of LLMs by interleaving it with search has been done before; most people call it "expert iteration" [3] (authors: please correct me if you think there is a *significant* difference between your method and this).
Especially what you call "SAFE_0" feels a bit rich: unless I am mistaken, you are literally just doing synthetic data generation with GPT-4o and filtering the results based on some measure of quality.
Also, on line 359 you say "21,398 [programs] have been successfully transformed [...] by SAFE"; unless I'm mistaken you mean "by GPT-4" here, because at this point you haven't done anything other than asking GPT to first transpile the code to Rust and to then transpile the Rust code into the subset of the language that is supported by Verus.

I would encourage the authors to tone down the language a bit more and focus on the actually novel parts of this paper, which I believe to be: the task target; finetuning on repair tasks to improve generation performance; and the metric used to filter the specification samples.

A more important issue is that I do not think the comparison to the baselines is fair in its current form for the "SAFE+" method.
The authors themselves point out that in this variation (i.e., when you do a round of self-debugging if the initial generation does not succeed), they generate `k * k` repair samples - how can you then compare against pass@1? You have actually drawn `k + k*k` samples from the model, so you should at least compare against a baseline of `pass@(k + k*k)`.
This is an issue that has come up again and again the self-debugging/refinement literature, and I once again encourage the authors to engage with that literature.
You still have good results here - for example, the SAFE+ pass@10 is substantially higher than the SAFE pass@100 on VerusBench - but the way you're currently presenting them overstates their significance.

Finally, the writing could use some more proof reading, especially the abstract and the introduction (but this is a minor complaint).


[3] 
@misc{polu2022formalmathematicsstatementcurriculum,
      title={Formal Mathematics Statement Curriculum Learning}, 
      author={Stanislas Polu and Jesse Michael Han and Kunhao Zheng and Mantas Baksys and Igor Babuschkin and Ilya Sutskever},
      year={2022},
      eprint={2202.01344},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2202.01344}, 
}

### Questions
- In table 2 you report a "total" column; I noticed the numbers don't add up if you just take the mean of the other columns, so I presume what you're actually doing is taking the mean over all of the samples in the entire dataset? (I think that's what you want to do, I just want to make sure I understood correctly).
- When constructing training data for the repair task, are you doing any filtering to make sure that the "incorrect program" is actually similar to the "correct program", or could they be completely different?
- What is the difference between SAFE and expert iteration, other than your synthetic data generation for the specifications?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The presented paper proposes a method to bootstrap training data for generating proofs of Rust code using LLMs. The pipeline starts from a small set of curated programs and gradually evolves using the verifier as signal for dataset curation. Finally they evaluate the resulting fine-tuned LLM and show state of the art results on a difficult dataset of low-resource correctness proofs.

### Strengths
- The paper is well-written and nicely structured. The figures and tables are well formatted and legible.
- The story is engaging and the tackled topic highly relevant.
- The results are clearly presented and provide interesting insights.

### Weaknesses
- In Table 1 the Accuracy of GPT-4o on VerusBench @100 is unfortunately missing (likely due to high inference cost?). Similarly the result of DeepSeekCoder RAW @100 is missing. If the authors could provide these values, the tables would provide a much more complete picture.
- In Table 2, Round 3 appears to severely degrade performance of the resulting model on the Tutorial dataset. Does this constitute some first signs of overfitting or collapse or could the authors provide some more insight on what is happening here? It might be interesting to provide some basis on deciding where to stop the iterative process.
- There is no discussion of Limitations. While the provided method is clearly powerful some discussion on potential limitations would be highly appreciated.

### Questions
Please provide a short statement or clarification to the points raised above.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SAFE, an innovative framework designed to address the challenges of automated proof generation for Rust code, SAFE overcomes the significant data scarcity issue ( i.e., there is far less proof data than code data for training language models) by using a self-evolving approach to synthesize specification/proof data and fine-tune models iteratively. SAFE operates through a feedback loop where synthesized proofs are continuously verified by a symbolic verifier, distinguishing correct from incorrect proofs. Incorrect proofs are used to train the model's self-debugging ability, while the correct proofs are used to improve the model for the next round. The design of the approach is smart and uses the insight that (1) using a quantitative metric to select high-quality specifications for fine-tuning; (2) we only need reasonably well, instead of perfect specifications to fine-tune in the next step; and (3) Verus can quickly tell correct proof from incorrect ones, which enables the collecting and filtering of large amount of data.

SAFE achieves a substantial improvement, attaining a 70.50% accuracy on a benchmark set crafted by human experts, a notable advancement over GPT-4o's performance of 24.46%. SAFE also obtains self-debugging ability using the incorrect proofs collected during the data collection step. Experiments show that each round of self-evolving improves the accuracy of SAFE, and proves the importance of using high-quality training data.

### Strengths
1. This paper proposes a novel approach that use self-evolving to iteratively improve LLM's ability of generating Rust specification and proofs. The fact that this approach does not rely on larger LLM such as GPT-4o in the following iterations (except the first round) makes it more generalizable and scalable.
2. The proposed approach shows great effectiveness, with three round of self-evolving, the fine-tuned LLM shows about 40% higher accuracy@1 compared to the prompting approach.
3. Comprehensive analysis and experiments, showing that each round of 1, 2 and 3 brings some improvement to the fine-tuned LLM (although the round 2 model is better than the round 3 model under some settings), and showing that high-quality specifications important to improve the model's accuracy during self-evolving.

### Weaknesses
1. The self-debugging ability is shown to be only effective for the first time, what could be potential approach for improving the self-debugging ability in the following rounds?
2. I am wondering if this self-evolving approach can improve smaller LLMs ability. For instance, if the backbone is DeepSeekCoder-1.3B, how effective is the self-evolving approach?

### Questions
1. A clarifying question about the self-evolving data: The data collected through GPT-4o (round 0) is used to fine-tuned the first specification/proof generation model. What's the data input used to let the generation model generate data for then next round? 
* Are these data the same programs as those used in the generating round 0 data? If this is the case, would the training data in each round kind of repetitive and lack of diversity?
* Or does author use some strategies to leave some unique programs for each round, so that the fine-tuning data for each round contains different programs?
2. Self-debugging is quite effective and improves the accuracy, how does the model obtain the ability of self-debugging? does the fine-tuning procedure contains self-debugging training data?
3. Why are the baseline models prompted with 4 examples instead of more examples?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes SAFE, a data generation and fine-tuning procedure for improving LLMs in generating proofs for the correctness of Rust code. SAFE consists of three stages: (i) verus-compatible code generation, (ii) self-evolving specification synthesis, and (iii) self-evolving proof synthesis. During stage (ii), SAFE leverages a symbolic and quantitative measure based on the correctness and completeness of the specification. For stage (iii), SAFE fine-tunes both proof generation and repair models. The experiments demonstrate the advantages of SAFE: it significantly improves the performance, compared to both the base model and GPT-4o.

### Strengths
This paper studies automating proof generation in formal program verification with LLMs, an important direction with great potential for practical applications. The focus is on Rust, a relatively new language that is gaining widespread adoption. Although synthetic data generation for fine-tuning LLMs is not a completely novel idea, the paper introduces a few interesting techniques for the domain of proof generation for Rust. I particularly like the metric for filtering high-quality specifications. The evaluation is thorough, demonstrating the benefits of SAFE over baselines and the effectiveness of its individual components.

### Weaknesses
The paper only focuses on small programs in the style of MBPP and CodeNet. Although I understand this is partly due to the limitation of the Verus tool, I do believe that the paper should present some case studies or discussion on how to scale the approach to real-world software projects.

Apart from proof generation, a major part of formal verification is writing the specifications. The paper covers mechanisms to fine-tune a “good” specification generation. It would strengthen the paper if more evaluation can be done on the specification generation task and how it can be combined with proof generation to automate end-to-end verification.

The paper lacks a study on the choice of the correctness and completeness thresholds for the specification metric.

The paper writing can be improved. Below are some issues I found or some recommendations:
- The text in Section 3 is sometimes ad-hoc and contains low-level details (e.g., choice of parameters). I would be helpful to revise the text to be more formal and move the details to later sections.
- Line 289: The paper says “much previous work relies on running many test cases” without providing any references.
- Line 519: Table 2 should be Table 3
- Table 3: The split of model names to multiple lines is confusing. I thought one line of text corresponds to one single baseline. The $\Delta$ rows look redundant as well.

### Questions
Please address the points raised in the “Weakness” section.

### Soundness
3

### Presentation
2

### Contribution
3
