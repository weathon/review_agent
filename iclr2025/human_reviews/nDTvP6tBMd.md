## Human Reviewer 1

### Summary
This work presents a new math dataset that targets on applied mathematics and requires graduate level knowledges. The problems in this dataset demand a combination of mathematical reasoning, computational tools, and subjective judgment. For example, it may need checks for self-consistency and the use of numerical methods.

### Strengths
1. This paper presents a method to generate graduate level math problems (in some pre-defined areas) and corresponding solutions.

### Weaknesses
1. The type of the math problems in this dataset is limited. First, only three main subjectives are discussed, ODEs, integrals and polynomial. Second, most of the problems are related to calculations, and it seems none of them is proof related. For such problems, the existed numerical computational software should be able to solve most of them. In summary, the diversity of this dataset is limited.

2. Since the dataset can be generated with python tools, it is possible to generate more examples for training. I didn't see any experiment about training on such dataset.

### Questions
1. The description in Section 3.2 is not clear. For example, the paper presents that `sympy` and `scipy` are used to implement the mathematical procedures required for obtaining approximate, analytical solutions. In Appendix A, it only includes some examples with questions and solutions, and I did not see the role of `sympy` and `scipy`.

2. As mentioned in Weakness 2, besides constructing the benchmark, I think this work has the potential to be used for constructing large scale training dataset, teaching the LLMs to solve such hard math problems with the suggested thoughts in CoT, or generated `sympy` code in PoT. What do the authors think about this direction?

3. As my concern of diversity in the Weakness 1, I also have the same concern on the solution ideas. It seems that many problems share the same solution idea by just randomly changing the numbers in the problem. This may be verified by the large performance increase from 0-shot to 5-shot. As the 5 demonstration examples may have already provide the correct solution path, the LLMs only need to change some numbers. Can the author introduce how many solution strategies are used for each type in your dataset?

### Soundness
3

### Presentation
3

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper proposes a new math benchmark for evaluation of LLMs.

### Strengths
This paper makes an important contribution to the LLM community. The LLM community needs a harder math benchmark with the growing capability of the models. I think this benchmark will be useful.

### Weaknesses
I don't see obvious weaknesses.

### Questions
Have the authors evaluated larger open-source models such as Llama 3.1 405B?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
1

---

## Human Reviewer 3

### Summary
The author introduces a challenging math dataset focused on asymptotic reasoning, highlighting the use of mathematical approximations to address complex problems that commonly arise in real-world scenarios.

### Strengths
1- Introducing a dataset that requires approximate analytical solutions is innovative, addressing a gap in current benchmarks. This approach aligns with many real-world problems and reflects how scientists typically tackle them.

2- The dataset is larger than similar graduate-level math datasets. It can be utilized to develop novel prompting techniques or for fine-tuning models.

### Weaknesses
The authors suggest that their dataset could be utilized for fine-tuning LLMs to enhance their capabilities. However, it would be valuable to see empirical results demonstrating this improvement.

The authors explore one-shot and five-shot CoT prompting, noting a substantial performance increase across most models. It would be worthwhile to investigate more complex CoT prompting techniques

### Questions
Refer to the weakness of paper section.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
6

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper proposes a dataset of mathematical problems taken from a graduate level engineering course, and evaluates the ability of several LLMs on solving the problems in the dataset. The problems include topics related to polynomials, root finding, integrals, etc. Paper considers some base problems and modifies them randomly to generate a relatively large dataset. The dataset also includes some manually crafted problems. The accuracy of best LLM goes above 62% in solving the problems in the test set of this dataset. The paper goes on to analyze the failures of LLMs.

### Strengths
The topic is interesting and the dataset (at least, a good portion of it) can be useful for the community.

Paper is well written and the experiments are insightful.

### Weaknesses
In my view, the dataset could have consisted of more difficult problems. From what I read in the abstract and introduction, I was expecting a much harder set of problems to be included in the dataset. The relatively high accuracy of the GPT model on the dataset (above 60%) is also indicative that a considerable portion of the dataset consists of problems that are not as challenging.

Dataset generation method, described in Section 3.2, is still not completely clear to me, even after reading Appendix A. I think the generation procedures need to be explained in more detail such that one can reproduce the generation method by reading the paper and the appendices. Appendix A merely provides some examples of problems in the dataset which I was hoping to see at least one sample in page 1 or 2, rather than the appendix.

The topic of non-dimensionalizing of a polynomial does not strike me as interesting as some of the other topics. In the appendix A, paper provides exact formulas for non-dimensionalizing a polynomial - an exact formula does not need numerical methods as the paper suggests are the basis of all problems in the dataset. Quoting the claim from page 1: “most datasets focus on grade school- to high school-level mathematics problems whose solution methods only involve direct, ‘clean’ calculations. In contrast, HARDMATH targets applied mathematics problems that require approximate analytical solutions.”

When it comes to the so-called Word problems in the dataset, the paper crafts them manually, which is fine. But, it does not sit well with earlier claims of the paper such as “Rather than relying on the typical approach of collecting problems from textbooks, standardized tests, or competitions, as seen in most existing datasets, we developed algorithms to automatically generate problems and their step-by-step solutions.”

Randomly modifying the initial conditions of an ODE or coefficients of a polynomial can of course be done automatically. However, I think it is not fair to contrast such automation with the manual work that was the basis for some of the benchmarks in the literature.

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
4