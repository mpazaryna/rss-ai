# 🇨🇿 BenCzechMark - Can your LLM Understand Czech?

**Source**: HuggingFace
**Date**: time.struct_time(tm_year=2024, tm_mon=10, tm_mday=1, tm_hour=0, tm_min=0, tm_sec=0, tm_wday=1, tm_yday=275, tm_isdst=0)

No summary available

[Read more](https://huggingface.co/blog/benczechmark)


## Full Content

🇨🇿 BenCzechMark - Can your LLM Understand Czech?
Hugging Face
Models
Datasets
Spaces
Posts
Docs
Solutions
Pricing
Log In
Sign Up
Back to Articles
🇨🇿 BenCzechMark - Can your LLM Understand Czech?
Published
October 1, 2024
Update on GitHub
Upvote
14
+8
mfajcik
Martin Fajčík
BUT-FIT
hynky
Hynek Kydlicek
mdocekal
Martin Dočekal
BUT-FIT
xdolez52
Jan Doležal
BUT-FIT
jstetina
Jakub Štětina
guest
Lakoc
Alexander Polok
BUT-FIT
popelucha
Zuzana Nevěřilová
guest
hales
Ales Horak
MU-NLPC
michal-stefanik
Michal Štefánik
MU-NLPC
Adamiros
Adam Jirkovský
CIIRC-NLP
JanH
Jan Hula
guest
jsedivy
Jan Sedivy
CIIRC-NLP
📋 Tasks and Categories
⚔️ Model Duels and Average Score
👑 BenCzechMark Leaderboard - Llama-405B Takes the Crown
🇨🇿 Think Your Model Can Excel in Czech? Submit It!
🌟 Acknowledgements
📚 Citation and references
The 🇨🇿 BenCzechMark is the first and most comprehensive evaluation suite for assessing the abilities of Large Language Models (LLMs) in the Czech language. It aims to test how well LLMs can:
Reason and perform complex tasks in Czech.
Generate and verify grammatically and semantically correct Czech.
Extract information and store knowledge by answering questions about Czech culture and Czech-related facts.
Do what language models were originally trained for—estimate the probability of Czech texts.
To achieve this, we've sourced
50
tasks spanning
9
categories, with 90% of tasks having native, non-translated content.
In this blog, we introduce both the evaluation suite itself and the BenCzechMark leaderboard, featuring over
25
open-source models of various sizes!
📋 Tasks and Categories
The 🇨🇿 BenCzechMark (in it’s current version) is divided into
9
categories to comprehensively assess LLM abilities. For each task,
We manually design at least 5 prompts, and record best performance and variance across prompts.
We distinguish between 4 types of tasks, and associate them with metrics:
Accuracy
(Acc) measures multi-choice(MC) tasks,
Exact Match
(EM) measures tasks with open short answer generation,
Area Under the Receiver Operating Characteristic Curve
(AUROC, computed as average of one-vs-all in multi-class setting) measures the performance on classification tasks, without need for threshold calibration.
Out-of-the-box language models are often biased by the class distributions in their training data, the way prompts are structured, and the examples provided during inference. These biases can vary across models, making predictions inconsistent depending on the specific model and its influences. To ensure reliable decision-making on datasets with different class distributions, calibration is necessary to adjust the model's predictions. However, by using threshold-free metrics like AUROC, which focus on ranking rather than decision thresholds, calibration can be avoided entirely. This approach enables fairer model comparisons by eliminating the need for calibration (see e.g.,
Zhaeo et al., 2021
for more details on calibration of LLMs).
Word-level Perplexity
(Ppl) is associated with language modeling tasks. It quantifies the likelihood the model would generate text with, normalized per number of words in corpus.
The translated portion of the dataset (10% of the total) was mostly translated via CUBBITT
LINDAT Translation
, except for
CsFever
, where the authors used
DeepL
for translation.
This is the complete list of categories, alongside the datasets and metrics used:
Reading Comprehension
tests whether the system can extract the answer for a question based on information provided in the context.
Belebele
- Acc - contains questions about manually translated web articles.
SQAD3.2
- EM -  is a well-established reading comprehension task in SQuAD format, sourced from Wikipedia.
Factual Knowledge
contains questions testing factual knowledge stored in the model.
Umimeto
(5 tasks focused on Biology/Chemistry/History/Informatics/Physics) - Acc - Elementary and high school questions from respective topics. Sourced from
umimeto.org
.
TriviaQA
- EM (Translated using CUBITT) - contains Q/A from trivia and quiz-league websites (U.S. centric dataset).
NaturalQuestions
- EM (Translated using CUBITT) -  contains Q/A from Google Search (U.S. centric dataset). We include these to ensure the model did not forget any EN-centric knowledge when prompted in Czech (i.e., after possible domain transfer).
Czech Language Understanding
targets the peculiar understanding of syntactic structure and nuanced meaning in the Czech Language.
CERMAT
(Open/TF/MC) - EM/AUROC/Acc  - focuses on understanding tasks sourced from 6th, 9th-year primary school tests and state high school exams in Open/True-False/Multiple-choice formats.
Grammar Error Detection
- AUC (True/False grammar error prediction task) - contains sentences from language learner essays.
Agree
- Acc - requires filling in missing grammar suffixes of past tense verbs
Language Modeling
tests how likely the model would sample specific Czech language samples.
Czech National Corpus
- Ppl - includes 7 tasks that span across spoken, dialect, historical, and other versions of Czech language, sourced from
ČNK
.
HellaSwag
- Acc - (Translated using CUBITT) requires selecting plausible continuation of text from 4 options.
Math Reasoning in Czech
quantifies how well the model can process and solve Czech math assignments.
Klokan QA
- Acc - elementary/high school problems from Czech math competition.
CERMAT
- EM/Acc - Math subsection of CERMAT Open/MC.
Umimeto (Math)
- Acc - Math subsection of Umimeto.
Natural Language Inference
tests whether the text entails the information required in the associated text pair.
Czech SNLI
- AUROC (Translated SNLI using CUBITT + manual correction) - tests for entailment of hypothesis in the premise text.
CSFever
- AUROC (Czech version of FEVER dataset, using partial translation) - asks whether claim is (at least partially) supported in the evidence.
CTKFacts
- AUROC- same format as CSFEVER, but manually sourced from Czech News Agency articles.
Propaganda
- AUROC - contains 13 tasks predicting various aspects of news articles, such as location, genre  and emotive theme.
Named Entity Recognition
determines whether the model recognizes different named entity types in the text.
CNEC2.0
- EM - standard NER dataset in Czech
Court Decisions
- EM - NER derived from decisions of Czech Supreme/Constitutional Courts.
Sentiment Analysis
quantifies how well the model estimates sentiment information in the text.
Subjectivity
- AUROC - asks whether a passage is subjective or objective.
CzechSentiment
(MALL/CSFD/FB) - AUROC - sentiment analysis of product reviews, movie reviews, and Facebook comments.
Document Retrieval
focuses on identifying the relevant documents.
Historical IR
- Acc - multiple-choice task for selecting passages relevant/irrelevant to a query.
⚔️ Model Duels and Average Score
Since we use different metrics for the tasks, simply averaging wouldn't work due to varying scales. Instead, we've introduced a novel way to determine a final score: we let the models fight!
For every task and metric, we compute a test for statistical significance at
α=0.05
. This means the probability that the performance of model A equals that of model B is estimated to be less than 0.05. We use the following tests, each with varying statistical power:
ACC and EM
: one-tailed paired t-test,
AUROC
: Bayesian test inspired by
Goutte et al., 2005
,
Ppl
: bootstrapping.
We then compute a model's
duel win score (DWS)
- the proportion of duels won against all other models on that task. Finally, we calculate aggregate scores as follows:
Category DWS: average of task scores within the category,
Average DWS: average across category DWSs.
This yields an easy-to-understand model score:
Macro-averaged model win-rate!
👑 BenCzechMark Leaderboard - Llama-405B Takes the Crown
To identify the top-performing open-source model in our suite, we evaluated
26 open-weight
models using the following parameters:
Maximum input length: 2048 tokens
Few-shot examples: 3
Truncation: Smart truncation (truncates few-shot samples first then task description)
Log-probability aggregation: Average-pooling (helps mitigate long-document bias)
Chat templates: Not used
The results can be explored in our
Space
. While Llama-450B emerged as the clear overall winner, it didn’t dominate every category. Interestingly, some models have excelled in specific areas — for instance:
Qwen-72B
shone in Math and Information Retrieval but lagged behind similarly-sized models in other categories.
Aya-23-35B
model excels in Sentiment and Language Modeling, but similarly lags behind in different categories.
Gemma-2 9B
delivers excellent results in Czech reading comprehension, outperforming much larger models.
🇨🇿 Think Your Model Can Excel in Czech? Submit It!
One of our main goals at
BenCzechMark
is to empower researchers to assess their models' capabilities in Czech and to encourage the community to train and discover models that excel in the Czech language.
If you know of a model that stands out, we'd love for you to
submit
it to our leaderboard, making the competition even more exciting!
To help you get started, we've prepared a straightforward 3-step guide, which you can find in the BenCzechMark space under the
Submission
tab.
🌟 Acknowledgements
We'd like to extend our thanks to all contributors from
BUT
FIT
,
FI
MUNI
,
CIIRC
CTU
, and
Hugging
Face
for their invaluable work in bringing BenCzechMark to life.
We're also grateful to the organizations that provided source data for some of the tasks, namely
Umímeto
,
CERMAT
, and
ČNK
.
📚 Citation and references
@article{fajcik2024benczechmark,
title = {{B}en{C}zech{M}ark: A Czech-centric Multitask and Multimetric Benchmark for Language Models with Duel Scoring Mechanism},
author = {Martin Fajcik and Martin Docekal and Jan Dolezal and Karel Ondrej and Karel Benes and Jan Kapsa and Michal Hradis and Zuzana Neverilova and Ales Horak and Michal Stefanik and Adam Jirkovsky and David Adamczyk and Jan Hula and Jan Sedivy and Hynek Kydlicek},
year = {2024},
url = {[https://huggingface.co/spaces/CZLC/BenCzechMark](https://huggingface.co/spaces/CZLC/BenCzechMark)}
institution = {Brno University of Technology, Masaryk University, Czech Technical University in Prague, Hugging Face},
}
More Articles from our Blog
Llama can now see and run on your device - welcome Llama 3.2
By
merve
September 25, 2024
•
139
Fine-tuning LLMs to 1.58bit: extreme quantization made easy
By
medmekk
September 18, 2024
•
146
Upvote
14
+2
Company
© Hugging Face
TOS
Privacy
About
Jobs
Website
Models
Datasets
Spaces
Pricing
Docs