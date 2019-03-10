Intuitive explanations of TF-IDF VSM, LSA, and context word embeddings in R code
================

-   [Summary](#summary)
-   [What is this document for?](#what-is-this-document-for)
-   [Where can I learn more about NLP?](#where-can-i-learn-more-about-nlp)
-   [An example of TF-IDF VSM](#an-example-of-tf-idf-vsm)
    -   [Documents to be searched](#documents-to-be-searched)
    -   [Pre-processing the text](#pre-processing-the-text)
    -   [Create the Term-Document Matrix](#create-the-term-document-matrix)
    -   [Calculate IDF for each Word across all documents](#calculate-idf-for-each-word-across-all-documents)
    -   [Calculate the weight for each word in each document](#calculate-the-weight-for-each-word-in-each-document)
    -   [Compare query to all documents](#compare-query-to-all-documents)
    -   [Unit vectors to compare word similarity regardless of frequency](#unit-vectors-to-compare-word-similarity-regardless-of-frequency)
-   [Beyond TF-IDF - Latent Semantic Analysis (LSA)](#beyond-tf-idf---latent-semantic-analysis-lsa)
-   [Word context - going beyond count based word embeddings - Word2vec](#word-context---going-beyond-count-based-word-embeddings---word2vec)

Summary
=======

1.  We learn what the TF-IDF VSM and then LSA are intuitively. TF-IDF VSM used to find which text documents are similar to each other in rank order. The highest ranking documents tend to have matching words that are **rarely** used across all documents.
2.  TF-IDF VSM can sound intimidating because of the the technical language used. "Term" means word, "Frequency" means count. The "Vector Space Model" is the part of the calculation that finds the angle between lists of matching word weights. This angle measures the similarity of documents.
3.  A very simple example is explained. It assumes no Maths or Natural Language Processing (NLP) knowledge. We use basic maths and simple cosine trigonometry.
4.  TF-IDF VSM is calculated as follows:
    1.  **Term Frequency (TF):** We first count how often each word is used within each document.
    2.  **Inverse Document Frequency (IDF):** We count the number of documents that contain each word. Inverse means dividing the total number of documents by the number of documents each word occurs in. The division gives common words a low IDF value, and rare words a high IDF value.
    3.  **TF-IDF:** For each word in each document, we multiply the word count in that document by the log of the TF multiplied by the IDF. Taking the log dampens the importance of the most rare words that are not as important as their raw untransformed value would indicate.
    4.  **Words embedded as vectors:** We call the lists of weights for each word in each document, vectors. When we line up the vectors for each document next to each other we call it a matrix. Putting words into such a matrix is a form of word embedding.
    5.  **Vector Spae Model (VSM):** We measure the similarity of each document vector against every other document vector one-by-one by measuring the angle between them. This is called the cosine similarity. We measure vector similarity by angle instead of distance to compensate for documents of varying lengths.
5.  We can sometimes improve the detection of document similarity using Latent Semantic Analysis (LSA). "Latent" means hidden, "Semantic" is meaning (i.e. hidden meaning analysis). In LSA, a document is given some of the information value from words **not** inside the document, but, those words are found inside documents that are similar to them. This is done by manipulating the matrix of word counts using Singular Value Decomposition (SVD).
6.  SVD is well worth learning intuitively too since it a fundamental technique behind many key Data Science tools such as data dimension reduction prior to Machine Learning, Principal Components Analysis (PCA) and solving linear equations.
7.  TF-IDF VSM and SVD are called count based methods. Modern methods use the context of words such as Word2vec.

What is this document for?
==========================

This document describes the TF-IDF VSM and LSA in plain English with simple examples so that you can understand it **intuitively**. A deeper intuitive understanding helps you move on to understand more complex NLP techniques such as Latent Semantic Analysis. Deeper understanding can also help you better identify where techniques are weak and strong. The [betterexplained](https://betterexplained.com/articles/adept-method/) website and the [Feynman Technique](https://medium.com/taking-note/learning-from-the-feynman-technique-5373014ad230) are two inspirations for explaining important techniques intuitively. In this spirit, this document does not assume any previous Maths or Natural Language Processing (NLP) knowledge.

Below are two R code conversions. The first is of a TF-IDF VSM example from this [tutorial](http://www.minerazzi.com/tutorials/term-vector-3.pdf) (page 6). It is logically followed by an R code conversion of the example in an [Introduction to Latent Semantic Analysis](http://lsa.colorado.edu/papers/dp1.LSAintro.pdf). The LSA example the tutorial uses is from the canonical [Indexing by Latent Semantic Analysis](http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf) paper from 1990.

Where can I learn more about NLP?
=================================

The calculations used in the code are not intended for use in a real project. For a real text mining project use a popular NLP package in R or Python, for example: - [tidytext](https://github.com/juliasilge/tidytext) - [quanteda](https://quanteda.io) - [text2vec](http://text2vec.org/index.html) - [scikit-learn](https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html).

Also, the following are excellent NLP tutorials taking you from basic to advanced knowledge, mostly assuming no prior knowledge:

-   [Speech and Language processing book](https://web.stanford.edu/~jurafsky/slp3/) and associated [YouTube videos](https://www.youtube.com/playlist?list=PLQiyVNMpDLKnZYBTUOlSI9mi9wAErFtFm),
-   [Introduction to Information Retrieval](https://nlp.stanford.edu/IR-book/)
-   [NLP-Guidance](https://moj-analytical-services.github.io/NLP-guidance/) written by Sam Tazzyman in the MoJ.

An example of TF-IDF VSM
========================

Documents to be searched
------------------------

In this example we use the search terms "gold silver truck" to find the document in Table 1 that matches most closely.

``` r
library(tidyverse)
library(magrittr)
library(kableExtra)

d1 <- c("Shipment of gold damaged in a fire.")
d2 <- c("Delivery of silver arrived in a silver truck.")
d3 <- c("Shipment of gold arrived in a truck.")
q1 <- c("gold silver truck")

query <- data.frame(q1) %>%
  tidyr::gather(key = "document", value = "text")

words <- data.frame(d1, d2, d3) %>%
  tidyr::gather(key = "document", value = "text")

kable_table(words, "Table 1: Three documents we want to search")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 1: Three documents we want to search
</caption>
<thead>
<tr>
<th style="text-align:left;">
document
</th>
<th style="text-align:left;">
text
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
Shipment of gold damaged in a fire.
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
Delivery of silver arrived in a silver truck.
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
Shipment of gold arrived in a truck.
</td>
</tr>
</tbody>
</table>

Pre-processing the text
-----------------------

We first use the [Tidytext::unnest\_tokens()](https://www.tidytextmining.com/tidytext.html#the-unnest_tokens-function) function to split the text for each document into one word per row. We then use [dplyr count](https://suzan.rbind.io/2018/04/dplyr-tutorial-4/#counting-the-number-of-observations) to count how often each word appears in each document.

``` r
library(tidytext)

seperate_words <- words %>%
  tidytext::unnest_tokens(word, text) %>%
  dplyr::count(document, word, sort = TRUE) %>%
  dplyr::ungroup() %>%
  dplyr::arrange(word, document)

kable_table(seperate_words, "Table 2: All words split out from each document")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 2: All words split out from each document
</caption>
<thead>
<tr>
<th style="text-align:left;">
document
</th>
<th style="text-align:left;">
word
</th>
<th style="text-align:right;">
n
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
damaged
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
delivery
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
fire
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
silver
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
1
</td>
</tr>
</tbody>
</table>

Create the Term-Document Matrix
-------------------------------

Next we reshape Table 2 from a long [Tidy](https://www.tidytextmining.com/tidytext.html#contrasting-tidy-text-with-other-data-structures) table into a wide table. The wide table is called a [Term-Document Matrix](https://moj-analytical-services.github.io/NLP-guidance/Glossary.html#tdm) where the terms are shown as one word per row, and each document is represented by its own column.

The last column *d*<sub>*i*</sub> counts in how many **documents** each word occurs one or more times across all documents. This is called the Document Frequency.

``` r
tdm <- seperate_words %>%
  tidyr::spread(key = document, value = n) %>%
  base::replace(is.na(.), 0) %>%
  dplyr::group_by_all() %>%
  dplyr::summarise(di = sum(d1 >= 1) + sum(d2 >= 1) + sum(d3 >= 1)) %>%
  dplyr::arrange(desc(di), word)

kable_table(tdm, "Table 3: Term Document Matrix (TDM)")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 3: Term Document Matrix (TDM)
</caption>
<thead>
<tr>
<th style="text-align:left;">
word
</th>
<th style="text-align:right;">
d1
</th>
<th style="text-align:right;">
d2
</th>
<th style="text-align:right;">
d3
</th>
<th style="text-align:right;">
di
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
</tr>
<tr>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
</tr>
<tr>
<td style="text-align:left;">
damaged
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
delivery
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
fire
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
silver
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
</tbody>
</table>

Calculate IDF for each Word across all documents
------------------------------------------------

The Inverse Document Frequency (IDF) value is a measure of the discriminatory power of each word globally across the collection of documents. The value is low for a word found in most documents, and high when it occurs in very few.

To calculate the *I**D**F*<sub>*i*</sub>, for each word, divide the total number of documents *D* (in this example it is 3) by the number of documents each word occurs in, *D*/*d*<sub>*i*</sub>. The most frequent words that occur in every document will equal 1 (3/3 = 1) and the most rare words occurring in only one document have the highest value of 3 (3/1 = 3).

Table 4 is sorted by *I**D**F*<sub>*i*</sub> in descending order. *I**D**F*<sub>*i*</sub> is the log of *D*/*d*<sub>*i*</sub>. Rare words like "damaged" have the highest *I**D**F*<sub>*i*</sub> as they appear in only one document. While words in all of the documents do not offer any discrimination. They have an *I**D**F*<sub>*i*</sub> value of zero (e.g. "of" = log(3/3) = 0). Low *I**D**F*<sub>*i*</sub> value words are often removed in the data preparation stage of text mining. These very common words are known as "stop words".

``` r
tdm <- tdm %>%
  mutate(
    Ddi = 3 / di,
    IDFi = base::log10(3 / di)
  ) %>%
  mutate_at(6:7, funs(round(., 2))) %>%
  dplyr::arrange(desc(IDFi), word)

kable_table(tdm, "Table 4: TDM with Inverse Document Frequency")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 4: TDM with Inverse Document Frequency
</caption>
<thead>
<tr>
<th style="text-align:left;">
word
</th>
<th style="text-align:right;">
d1
</th>
<th style="text-align:right;">
d2
</th>
<th style="text-align:right;">
d3
</th>
<th style="text-align:right;">
di
</th>
<th style="text-align:right;">
Ddi
</th>
<th style="text-align:right;">
IDFi
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
damaged
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
</tr>
<tr>
<td style="text-align:left;">
delivery
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
</tr>
<tr>
<td style="text-align:left;">
fire
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
</tr>
<tr>
<td style="text-align:left;">
silver
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
</tr>
<tr>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
</tbody>
</table>

Intuitively, the importance of a word in document will not increase proportionally with frequency. Therefore, a simple method to reduce the importance of rare words is to take the logarithm of the ratio of *D*/*d*<sub>*i*</sub>. This is particularly important for rare words used in only one document. They will have an IDF value the same as the total number of documents which can be in the thousands or more. A further exploration of the theory of IDF and how it relates to Information theory is described in detail [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.97.7340&rep=rep1&type=pdf)

Typically, log of base-10 is used for TF-IDF. Base-10 allows faster mental calculation. For example,log(1)=0, log(10)=1, log(100)=2, etc. It is also simple to transform *I**D**F*<sub>*i*</sub> back to the original ratio by raising 10 by that value. For example, the *I**D**F*<sub>*i*</sub> value for "damaged" can be returned to the raw *D*/*d*<sub>*i*</sub> ratio by raising 10 to the power of 0.4771213, (10<sup>0.4771213</sup> = 3).

Calculate the weight for each word in each document
---------------------------------------------------

We now search the collection of three documents with our search terms by adding the column called "query". For each word, we multiply its *I**D**F*<sub>*i*</sub> value by the Term Frequency, which is simply the number of times the word appears in both the query and in each document. This weights the word counts in each document and the query by the discriminatory information each word has been found to provide across all the documents globally.

Look again at Table 1. We can see "silver" is rare globally among all the documents. It is only used in document 2. While"silver" is a common word locally as it is used twice in document 2. No other word is used twice in the same document. Looking at the weights calculated for "silver", the importance of the word "silver" is reflected in the high weight calculated. Silver appears in *d*<sub>2</sub> two times, and when multiplied with its *I**D**F*<sub>*i*</sub> it has the value 0.954 shown in the column *w*<sub>*i*</sub>2. This is the highest weight in the whole matrix.

``` r
query_seperate_words <- query %>%
  tidytext::unnest_tokens(word, text) %>%
  dplyr::count(document, word, sort = TRUE) %>%
  dplyr::ungroup()

joined <- tdm %>%
  dplyr::left_join(query_seperate_words, by = "word") %>%
  dplyr::select(-document) %>%
  base::replace(is.na(.), 0) %>%
  dplyr::rename(query = n) %>%
  dplyr::mutate(
    wiq = query * IDFi,
    wi1 = d1 * IDFi,
    wi2 = d2 * IDFi,
    wi3 = d3 * IDFi
  ) %>%
  dplyr::arrange(desc(IDFi), word) %>%
  mutate_at(6:11, funs(round(., 2)))

kable_table(joined, "Table 5: Weight word frequecies with the IDF value for each word")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 5: Weight word frequecies with the IDF value for each word
</caption>
<thead>
<tr>
<th style="text-align:left;">
word
</th>
<th style="text-align:right;">
d1
</th>
<th style="text-align:right;">
d2
</th>
<th style="text-align:right;">
d3
</th>
<th style="text-align:right;">
di
</th>
<th style="text-align:right;">
Ddi
</th>
<th style="text-align:right;">
IDFi
</th>
<th style="text-align:right;">
query
</th>
<th style="text-align:right;">
wiq
</th>
<th style="text-align:right;">
wi1
</th>
<th style="text-align:right;">
wi2
</th>
<th style="text-align:right;">
wi3
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
damaged
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
delivery
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
fire
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
silver
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.96
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
<tr>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
</tbody>
</table>

Compare query to all documents
------------------------------

To compensate for the effect of [document length](https://cmry.github.io/notes/euclidean-v-cosine), the standard way of quantifying the similarity between the words in the query and the words in each document is to measure the angle between the two vectors of words. This is called the [cosine of similarity](https://nlp.stanford.edu/IR-book/html/htmledition/dot-products-1.html).

To find the angle between the vectors we multiply each TF-IDF weighted word frequency in the query by the IDF weighted word frequency for each document, then sum all the values. This multiplication is also called the dot product. The dot product is the sum of the products of each component of the two vectors.

This [example](https://www.varsitytutors.com/precalculus-help/find-the-measure-of-an-angle-between-two-vectors), shows that the dot product equals the product of the length of the two vectors and the [cosine of the angle](https://www.mathopenref.com/cosine.html), a.b = |a|.|b|.*c**o**s**θ* That is to say, the dot product of two vectors will be equal to the cosine of the angle between the vectors, times the lengths of each of the vectors.

The cosine similarity value shows that document 2 is the closest match to the search terms "gold silver truck". We can understand why intuitively if we consider that document 2 contains two of the search terms ("silver" and "truck"). Also, document 2 is the only document to contain the rare word "silver".

``` r
vector_length <- function(vec) {
  return(sqrt(sum(vec^2, na.rm = TRUE)))
}

cosine_similarity <- function(a, b) {
  sum(crossprod(a, b)) /
    (vector_length(a) * vector_length(b))
}

d1cosine <- cosine_similarity(joined$wiq, joined$wi1)
d2cosine <- cosine_similarity(joined$wiq, joined$wi2)
d3cosine <- cosine_similarity(joined$wiq, joined$wi3)

cosine <- c(d1cosine, d2cosine, d3cosine)
document <- c("document1", "document2", "document3")
text <- c(d1, d2, d3)
query <- c(q1, q1, q1)

which_document <- data.frame(document, text, query, cosine) %>%
  dplyr::arrange(desc(cosine)) %>%
  mutate_at(4, funs(round(., 2)))

kable_table(which_document, "Table 7: Cosine similarity value between the query and each document") 
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 7: Cosine similarity value between the query and each document
</caption>
<thead>
<tr>
<th style="text-align:left;">
document
</th>
<th style="text-align:left;">
text
</th>
<th style="text-align:left;">
query
</th>
<th style="text-align:right;">
cosine
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
document2
</td>
<td style="text-align:left;">
Delivery of silver arrived in a silver truck.
</td>
<td style="text-align:left;">
gold silver truck
</td>
<td style="text-align:right;">
0.82
</td>
</tr>
<tr>
<td style="text-align:left;">
document3
</td>
<td style="text-align:left;">
Shipment of gold arrived in a truck.
</td>
<td style="text-align:left;">
gold silver truck
</td>
<td style="text-align:right;">
0.33
</td>
</tr>
<tr>
<td style="text-align:left;">
document1
</td>
<td style="text-align:left;">
Shipment of gold damaged in a fire.
</td>
<td style="text-align:left;">
gold silver truck
</td>
<td style="text-align:right;">
0.08
</td>
</tr>
</tbody>
</table>

Unit vectors to compare word similarity regardless of frequency
---------------------------------------------------------------

Another way to calculate the cosine of the angle is to [normalise](https://moj-analytical-services.github.io/NLP-guidance/Glossary.html#norm) each vector into a "unit vector".

The reason for normalising is well explained by [Dan Jurafsky](https://web.stanford.edu/~jurafsky/slp3/6.pdf) page (11), *"The dot product is higher if a vector is longer, with higher values in each dimension. More frequent words have longer vectors, since they tend to co-occur with more words and have higher co-occurrence values with each of them. The raw dot product thus will be higher for frequent words. But this is a problem; we’d like a similarity metric that tells us how similar two words are regardless of their frequency."*

The normalised columns (or unit vectors) can be seen below in Table 6 as qhat, d1hat, d2hat, d3hat. Normalising a vector means converting the vector to a length of 1 by dividing each value by the vector length. We calculate length by taking the square root of the sum of all squared values in each vector.

``` r
unit_vector <- function(vec) {
  return(vec / (sqrt(sum(vec^2, na.rm = TRUE))))
}

joined$qhat <- unit_vector(joined$wiq)
joined$d1hat <- unit_vector(joined$wi1)
joined$d2hat <- unit_vector(joined$wi2)
joined$d3hat <- unit_vector(joined$wi3)

joined <- joined %>%
  mutate_at(6:15, funs(round(., 2)))

kable_table(joined, "Table 6: Converting weighted word frequenies into unit vectors ")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 6: Converting weighted word frequenies into unit vectors
</caption>
<thead>
<tr>
<th style="text-align:left;">
word
</th>
<th style="text-align:right;">
d1
</th>
<th style="text-align:right;">
d2
</th>
<th style="text-align:right;">
d3
</th>
<th style="text-align:right;">
di
</th>
<th style="text-align:right;">
Ddi
</th>
<th style="text-align:right;">
IDFi
</th>
<th style="text-align:right;">
query
</th>
<th style="text-align:right;">
wiq
</th>
<th style="text-align:right;">
wi1
</th>
<th style="text-align:right;">
wi2
</th>
<th style="text-align:right;">
wi3
</th>
<th style="text-align:right;">
qhat
</th>
<th style="text-align:right;">
d1hat
</th>
<th style="text-align:right;">
d2hat
</th>
<th style="text-align:right;">
d3hat
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
damaged
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.66
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
delivery
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
fire
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.66
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
silver
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3.0
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.48
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.96
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.88
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.87
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
arrived
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.5
</td>
</tr>
<tr>
<td style="text-align:left;">
gold
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.5
</td>
</tr>
<tr>
<td style="text-align:left;">
shipment
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.5
</td>
</tr>
<tr>
<td style="text-align:left;">
truck
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.5
</td>
</tr>
<tr>
<td style="text-align:left;">
a
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
in
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
<tr>
<td style="text-align:left;">
of
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
3
</td>
<td style="text-align:right;">
1.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
</tr>
</tbody>
</table>
Once normalised, the dot product of two vectors computes the cosine of the angle between them. This simple dot product calculation of normalised vectors in table 6 creates Table 7 below. We can see that documents 1 and 2 are most similar to each other with a value of 0.24 (Table 7).

``` r
joined_matrix <- as.matrix(joined[, 13:16])
compare_all <- base::crossprod(joined_matrix, joined_matrix)
colnames(compare_all) <- c("q", "d1", "d2", "d3")
rownames(compare_all) <- c("q", "d1", "d2", "d3")

kable_table(round(compare_all, 2), "Table 7: Compare all vectors with cosine similarity")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 7: Compare all vectors with cosine similarity
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
q
</th>
<th style="text-align:right;">
d1
</th>
<th style="text-align:right;">
d2
</th>
<th style="text-align:right;">
d3
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
q
</td>
<td style="text-align:right;">
0.99
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.82
</td>
<td style="text-align:right;">
0.33
</td>
</tr>
<tr>
<td style="text-align:left;">
d1
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.25
</td>
</tr>
<tr>
<td style="text-align:left;">
d2
</td>
<td style="text-align:right;">
0.82
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
1.00
</td>
<td style="text-align:right;">
0.16
</td>
</tr>
<tr>
<td style="text-align:left;">
d3
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
1.00
</td>
</tr>
</tbody>
</table>

Beyond TF-IDF - Latent Semantic Analysis (LSA)
==============================================

Representing documents as vectors is called embedding. We can sometimes "improve" how document words embedded in a matrix can find similar documents using Latent Semantic Analysis (LSA). "Latent" means hidden, "Semantic" is meaning (i.e. hidden meaning analysis). In LSA, a document is given some of the information value from words **not** inside the document, but those words are found inside documents that are similar to them.

LSA uses Singular Value Decomposition (SVD). It is well worth learning SVD intuitively too as it a fundamental technique behind many key Data Science tools:

-   Data dimension reduction prior to Machine Learning and Principal Components Analysis [PCA](https://stats.idre.ucla.edu/r/codefragments/svd_demos/)
-   [Image compression](https://towardsdatascience.com/singular-value-decomposition-with-example-in-r-948c3111aa43)
-   [Solving](http://www.math.usu.edu/~corcoran/classes/old/07spring6550/examples/svd.pdf) linear equations

Below we convert to R a clearly explained example from an [Introduction to Latent Semantic Analysis](http://lsa.colorado.edu/papers/dp1.LSAintro.pdf). It uses the example from [Indexing by Latent Semantic Analysis](http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf).

The example takes the following nine titles, five about human computer interaction (c1 to c5), and four about mathematical graph theory (m1 to m5).

-   c1: Human machine interface for ABC computer applications
-   c2: A survey of user opinion of computer system response time
-   c3: The EPS user interface management system
-   c4: System and human system engineering testing of EPS
-   c5: Relation of user perceived response time to error measurement
-   m1: The generation of random, binary, ordered trees
-   m2: The intersection graph of paths in trees
-   m3: Graph minors IV: Widths of trees and well-quasi-ordering
-   m4: Graph minors: A survey

First we calculate the Term Document Matrix (or count of words in each document).

``` r
# https://tutorials.quanteda.io/basic-operations/tokens/tokens_select/
txt <- c(c1 = "Human machine interface for ABC computer applications",
c2 = "A survey of user opinion of computer system response time",
c3 = "The EPS user interface management system",
c4 = "System and human system engineering testing of EPS",
c5 = "Relation of user perceived response time to error measurement",
m1 = "The generation of random, binary, ordered trees",
m2 = "The intersection graph of paths in trees",
m3 = "Graph minors IV: Widths of trees and well-quasi-ordering",
m4 = "Graph minors: A survey")

toks <- quanteda::tokens(txt)
toks_nostop <-quanteda::tokens_select(toks, c("human","interface","computer","user","system","response","time","EPS","survey","trees","graph","minors")
, selection = "keep", padding = FALSE)

mydfm <- quanteda::dfm(toks_nostop)

# Convert to a term document matrix to match the example
tdm <- base::t(as.matrix(mydfm))

kable_table(tdm, "Table 1: The Term Document Matrix")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 1: The Term Document Matrix
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
c1
</th>
<th style="text-align:right;">
c2
</th>
<th style="text-align:right;">
c3
</th>
<th style="text-align:right;">
c4
</th>
<th style="text-align:right;">
c5
</th>
<th style="text-align:right;">
m1
</th>
<th style="text-align:right;">
m2
</th>
<th style="text-align:right;">
m3
</th>
<th style="text-align:right;">
m4
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
human
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
interface
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
computer
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
survey
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
user
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
system
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
response
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
time
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
eps
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
trees
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
graph
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
minors
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
</tr>
</tbody>
</table>

We "decompose" the above matrix in Table 1 into three other matrices using the R base function [svd()](https://stat.ethz.ch/R-manual/R-devel/library/base/html/svd.html). This R function implements [Singular Value Decomposition](https://en.wikipedia.org/wiki/Singular_value_decomposition). The transformation, *"can be [intuitively interpreted](https://en.wikipedia.org/wiki/Singular_value_decomposition#Intuitive_interpretations) as a composition of three geometrical transformations: a rotation or reflection, a scaling, and another rotation or reflection."* If we multiply together the three decomposed matrices this exactly re-create the original matrix.

Table 2 is the result of decomposing Table 1 using SVD into U (the orthogonal matrix), D (the diagonal matrix), and V' (the transposed orthogonal matrix). *(Note if you compare Table 2 to the example in [Indexing by Latent Semantic Analysis](http://www.cs.bham.ac.uk/~pxt/IDA/lsa_ind.pdf) page 406, some of the signs are different. The reason for this ambiguity is explained [here](https://prod-ng.sandia.gov/techlib-noauth/access-control.cgi/2007/076422.pdf).)*

``` r
s <- base::svd(tdm)

u <- round(s$u,2)
d <- round(base::diag(s$d, 9, 9),2) # placing values on the diagonal
v <- round(base::t(s$v),2) # transpose the matrix (swap rows with columns)

knitr::kable(list(u,d,v), caption = "Table 2: U orthogonal matrix, D diagonal matrix, V' transposed orthogoanl matrix")
```

<table class="kable_wrapper">
<caption>
Table 2: U orthogonal matrix, D diagonal matrix, V' transposed orthogoanl matrix
</caption>
<tbody>
<tr>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
-0.22
</td>
<td style="text-align:right;">
-0.11
</td>
<td style="text-align:right;">
0.29
</td>
<td style="text-align:right;">
-0.41
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
0.34
</td>
<td style="text-align:right;">
0.52
</td>
<td style="text-align:right;">
-0.06
</td>
<td style="text-align:right;">
-0.41
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.20
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
0.14
</td>
<td style="text-align:right;">
-0.55
</td>
<td style="text-align:right;">
-0.28
</td>
<td style="text-align:right;">
-0.50
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
-0.11
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.24
</td>
<td style="text-align:right;">
0.04
</td>
<td style="text-align:right;">
-0.16
</td>
<td style="text-align:right;">
-0.59
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
-0.30
</td>
<td style="text-align:right;">
0.06
</td>
<td style="text-align:right;">
0.49
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.21
</td>
<td style="text-align:right;">
0.27
</td>
<td style="text-align:right;">
-0.18
</td>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
0.54
</td>
<td style="text-align:right;">
-0.08
</td>
<td style="text-align:right;">
-0.47
</td>
<td style="text-align:right;">
-0.04
</td>
<td style="text-align:right;">
-0.58
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.40
</td>
<td style="text-align:right;">
0.06
</td>
<td style="text-align:right;">
-0.34
</td>
<td style="text-align:right;">
0.10
</td>
<td style="text-align:right;">
-0.33
</td>
<td style="text-align:right;">
-0.38
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.01
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.64
</td>
<td style="text-align:right;">
-0.17
</td>
<td style="text-align:right;">
0.36
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
-0.17
</td>
<td style="text-align:right;">
0.03
</td>
<td style="text-align:right;">
0.27
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
-0.43
</td>
<td style="text-align:right;">
0.07
</td>
<td style="text-align:right;">
-0.08
</td>
<td style="text-align:right;">
0.17
</td>
<td style="text-align:right;">
0.28
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.05
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
-0.43
</td>
<td style="text-align:right;">
0.07
</td>
<td style="text-align:right;">
-0.08
</td>
<td style="text-align:right;">
0.17
</td>
<td style="text-align:right;">
0.28
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.05
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.30
</td>
<td style="text-align:right;">
-0.14
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
-0.11
</td>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.03
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.17
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
0.49
</td>
<td style="text-align:right;">
0.23
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
-0.59
</td>
<td style="text-align:right;">
0.39
</td>
<td style="text-align:right;">
-0.29
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
-0.23
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.04
</td>
<td style="text-align:right;">
0.62
</td>
<td style="text-align:right;">
0.22
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.07
</td>
<td style="text-align:right;">
-0.11
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
-0.68
</td>
<td style="text-align:right;">
0.23
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
0.14
</td>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
0.30
</td>
<td style="text-align:right;">
-0.28
</td>
<td style="text-align:right;">
0.34
</td>
<td style="text-align:right;">
0.68
</td>
<td style="text-align:right;">
0.18
</td>
</tr>
</tbody>
</table>
</td>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
3.34
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
2.54
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
2.35
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
1.64
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
1.5
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
1.31
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.85
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.56
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.0
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
0.36
</td>
</tr>
</tbody>
</table>
</td>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
-0.20
</td>
<td style="text-align:right;">
-0.61
</td>
<td style="text-align:right;">
-0.46
</td>
<td style="text-align:right;">
-0.54
</td>
<td style="text-align:right;">
-0.28
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.08
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.06
</td>
<td style="text-align:right;">
0.17
</td>
<td style="text-align:right;">
-0.13
</td>
<td style="text-align:right;">
-0.23
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
0.62
</td>
<td style="text-align:right;">
0.53
</td>
</tr>
<tr>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
-0.50
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
0.57
</td>
<td style="text-align:right;">
-0.51
</td>
<td style="text-align:right;">
0.10
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.08
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.95
</td>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
0.04
</td>
<td style="text-align:right;">
0.27
</td>
<td style="text-align:right;">
0.15
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
0.01
</td>
<td style="text-align:right;">
-0.02
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.05
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
-0.38
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
-0.33
</td>
<td style="text-align:right;">
-0.39
</td>
<td style="text-align:right;">
-0.35
</td>
<td style="text-align:right;">
-0.15
</td>
<td style="text-align:right;">
0.60
</td>
</tr>
<tr>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.26
</td>
<td style="text-align:right;">
-0.72
</td>
<td style="text-align:right;">
0.37
</td>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
0.30
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
-0.36
</td>
</tr>
<tr>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
-0.43
</td>
<td style="text-align:right;">
-0.24
</td>
<td style="text-align:right;">
0.26
</td>
<td style="text-align:right;">
0.67
</td>
<td style="text-align:right;">
-0.34
</td>
<td style="text-align:right;">
-0.15
</td>
<td style="text-align:right;">
0.25
</td>
<td style="text-align:right;">
0.04
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
0.05
</td>
<td style="text-align:right;">
0.01
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.06
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
-0.76
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
-0.07
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.06
</td>
<td style="text-align:right;">
0.24
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
-0.08
</td>
<td style="text-align:right;">
-0.26
</td>
<td style="text-align:right;">
-0.62
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
0.52
</td>
<td style="text-align:right;">
-0.45
</td>
</tr>
</tbody>
</table>
</td>
</tr>
</tbody>
</table>

To demonstrate that the three matrices U, D and V' above are a decomposition of the first, when multiplied together the result in Table 3 is the same as our original matrix in Table 1.

``` r
reconstruct <- round(u %*% d %*% v,0)

colnames(reconstruct) <- colnames(tdm)
rownames(reconstruct) <- rownames(tdm)

kable_table(reconstruct, "Table 3: Reconstructing our orginal matrix")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 3: Reconstructing our orginal matrix
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
c1
</th>
<th style="text-align:right;">
c2
</th>
<th style="text-align:right;">
c3
</th>
<th style="text-align:right;">
c4
</th>
<th style="text-align:right;">
c5
</th>
<th style="text-align:right;">
m1
</th>
<th style="text-align:right;">
m2
</th>
<th style="text-align:right;">
m3
</th>
<th style="text-align:right;">
m4
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
human
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
interface
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
computer
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
survey
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
user
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
system
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
2
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
response
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
time
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
eps
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
trees
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
0
</td>
</tr>
<tr>
<td style="text-align:left;">
graph
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
</tr>
<tr>
<td style="text-align:left;">
minors
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
0
</td>
<td style="text-align:right;">
1
</td>
<td style="text-align:right;">
1
</td>
</tr>
</tbody>
</table>
We can now reduce the dimensionality if we select only the first two dimensions of each matrix as shown in Table 4 below.

``` r
# find largest singular values
s_red <- RSpectra::svds(tdm,2)

u_red <- round(s_red$u,2)
d_red <- round(base::diag(s_red$d, 2, 2),2)
v_red <- round(base::t(s_red$v),2)

knitr::kable(list(u_red,d_red,v_red), caption = "Table 4: U, D and V' after selecting first two dimesions")
```

<table class="kable_wrapper">
<caption>
Table 4: U, D and V' after selecting first two dimesions
</caption>
<tbody>
<tr>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
-0.22
</td>
<td style="text-align:right;">
-0.11
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.20
</td>
<td style="text-align:right;">
-0.07
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.24
</td>
<td style="text-align:right;">
0.04
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.21
</td>
<td style="text-align:right;">
0.27
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.40
</td>
<td style="text-align:right;">
0.06
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.64
</td>
<td style="text-align:right;">
-0.17
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.11
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.11
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.30
</td>
<td style="text-align:right;">
-0.14
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
0.49
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.04
</td>
<td style="text-align:right;">
0.62
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
0.45
</td>
</tr>
</tbody>
</table>
</td>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
3.34
</td>
<td style="text-align:right;">
0.00
</td>
</tr>
<tr>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
2.54
</td>
</tr>
</tbody>
</table>
</td>
<td>
<table>
<tbody>
<tr>
<td style="text-align:right;">
-0.20
</td>
<td style="text-align:right;">
-0.61
</td>
<td style="text-align:right;">
-0.46
</td>
<td style="text-align:right;">
-0.54
</td>
<td style="text-align:right;">
-0.28
</td>
<td style="text-align:right;">
0.00
</td>
<td style="text-align:right;">
-0.01
</td>
<td style="text-align:right;">
-0.02
</td>
<td style="text-align:right;">
-0.08
</td>
</tr>
<tr>
<td style="text-align:right;">
-0.06
</td>
<td style="text-align:right;">
0.17
</td>
<td style="text-align:right;">
-0.13
</td>
<td style="text-align:right;">
-0.23
</td>
<td style="text-align:right;">
0.11
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
0.62
</td>
<td style="text-align:right;">
0.53
</td>
</tr>
</tbody>
</table>
</td>
</tr>
</tbody>
</table>

When we multiply the three reduced matricies in Table 4 this creates Table 5 below. You can see that while the word "trees" is not in the title of m4 ("Graph minors: A survey"), "trees" does now have some weight (0.66, Table 5). This is because "trees" is in a document that is very similar (m3 = "Graph minors IV: Widths of trees and well-quasi-ordering"). Also, the original value of 1.00 for "survey" in Table 1, which appeared once in m4, has been replaced by 0.42.

Describing this intuitively, *"in constructing the reduced dimensional representation, SVD, with only values along two orthogonal dimensions to go on, has to estimate what words actually appear in each context by using only the information it has extracted. It does that by saying: This text segment is best described as having so much of abstract concept one and so much of abstract concept two, and this word has so much of concept one and so much of concept two, and combining those two pieces of information (by vector arithmetic), my best guess is that word X actually appeared 0.6 times in context Y."* pages 12 & 14 of the [Introduction to Latent Semantic Analysis](http://lsa.colorado.edu/papers/dp1.LSAintro.pdf). \_

``` r
final <- round(u_red %*% d_red %*% v_red,2)

colnames(final) <- colnames(tdm)
rownames(final) <- rownames(tdm)

kable_table(final, "Table 5: Multiplication of reduced matriices U, D and V'")
```

<table class="table table-striped table-condensed" style="width: auto !important; ">
<caption>
Table 5: Multiplication of reduced matriices U, D and V'
</caption>
<thead>
<tr>
<th style="text-align:left;">
</th>
<th style="text-align:right;">
c1
</th>
<th style="text-align:right;">
c2
</th>
<th style="text-align:right;">
c3
</th>
<th style="text-align:right;">
c4
</th>
<th style="text-align:right;">
c5
</th>
<th style="text-align:right;">
m1
</th>
<th style="text-align:right;">
m2
</th>
<th style="text-align:right;">
m3
</th>
<th style="text-align:right;">
m4
</th>
</tr>
</thead>
<tbody>
<tr>
<td style="text-align:left;">
human
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
0.37
</td>
<td style="text-align:right;">
0.46
</td>
<td style="text-align:right;">
0.18
</td>
<td style="text-align:right;">
-0.05
</td>
<td style="text-align:right;">
-0.12
</td>
<td style="text-align:right;">
-0.16
</td>
<td style="text-align:right;">
-0.09
</td>
</tr>
<tr>
<td style="text-align:left;">
interface
</td>
<td style="text-align:right;">
0.14
</td>
<td style="text-align:right;">
0.38
</td>
<td style="text-align:right;">
0.33
</td>
<td style="text-align:right;">
0.40
</td>
<td style="text-align:right;">
0.17
</td>
<td style="text-align:right;">
-0.03
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
-0.10
</td>
<td style="text-align:right;">
-0.04
</td>
</tr>
<tr>
<td style="text-align:left;">
computer
</td>
<td style="text-align:right;">
0.15
</td>
<td style="text-align:right;">
0.51
</td>
<td style="text-align:right;">
0.36
</td>
<td style="text-align:right;">
0.41
</td>
<td style="text-align:right;">
0.24
</td>
<td style="text-align:right;">
0.02
</td>
<td style="text-align:right;">
0.05
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.12
</td>
</tr>
<tr>
<td style="text-align:left;">
survey
</td>
<td style="text-align:right;">
0.10
</td>
<td style="text-align:right;">
0.54
</td>
<td style="text-align:right;">
0.23
</td>
<td style="text-align:right;">
0.22
</td>
<td style="text-align:right;">
0.27
</td>
<td style="text-align:right;">
0.13
</td>
<td style="text-align:right;">
0.31
</td>
<td style="text-align:right;">
0.44
</td>
<td style="text-align:right;">
0.42
</td>
</tr>
<tr>
<td style="text-align:left;">
user
</td>
<td style="text-align:right;">
0.26
</td>
<td style="text-align:right;">
0.84
</td>
<td style="text-align:right;">
0.59
</td>
<td style="text-align:right;">
0.69
</td>
<td style="text-align:right;">
0.39
</td>
<td style="text-align:right;">
0.03
</td>
<td style="text-align:right;">
0.08
</td>
<td style="text-align:right;">
0.12
</td>
<td style="text-align:right;">
0.19
</td>
</tr>
<tr>
<td style="text-align:left;">
system
</td>
<td style="text-align:right;">
0.45
</td>
<td style="text-align:right;">
1.23
</td>
<td style="text-align:right;">
1.04
</td>
<td style="text-align:right;">
1.25
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
-0.08
</td>
<td style="text-align:right;">
-0.17
</td>
<td style="text-align:right;">
-0.22
</td>
<td style="text-align:right;">
-0.06
</td>
</tr>
<tr>
<td style="text-align:left;">
response
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.60
</td>
<td style="text-align:right;">
0.38
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
0.28
</td>
<td style="text-align:right;">
0.05
</td>
<td style="text-align:right;">
0.13
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
0.22
</td>
</tr>
<tr>
<td style="text-align:left;">
time
</td>
<td style="text-align:right;">
0.16
</td>
<td style="text-align:right;">
0.60
</td>
<td style="text-align:right;">
0.38
</td>
<td style="text-align:right;">
0.42
</td>
<td style="text-align:right;">
0.28
</td>
<td style="text-align:right;">
0.05
</td>
<td style="text-align:right;">
0.13
</td>
<td style="text-align:right;">
0.19
</td>
<td style="text-align:right;">
0.22
</td>
</tr>
<tr>
<td style="text-align:left;">
eps
</td>
<td style="text-align:right;">
0.22
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
0.51
</td>
<td style="text-align:right;">
0.62
</td>
<td style="text-align:right;">
0.24
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
-0.15
</td>
<td style="text-align:right;">
-0.20
</td>
<td style="text-align:right;">
-0.11
</td>
</tr>
<tr>
<td style="text-align:left;">
trees
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
0.23
</td>
<td style="text-align:right;">
-0.15
</td>
<td style="text-align:right;">
-0.27
</td>
<td style="text-align:right;">
0.15
</td>
<td style="text-align:right;">
0.24
</td>
<td style="text-align:right;">
0.55
</td>
<td style="text-align:right;">
0.77
</td>
<td style="text-align:right;">
0.66
</td>
</tr>
<tr>
<td style="text-align:left;">
graph
</td>
<td style="text-align:right;">
-0.07
</td>
<td style="text-align:right;">
0.35
</td>
<td style="text-align:right;">
-0.14
</td>
<td style="text-align:right;">
-0.29
</td>
<td style="text-align:right;">
0.21
</td>
<td style="text-align:right;">
0.30
</td>
<td style="text-align:right;">
0.69
</td>
<td style="text-align:right;">
0.98
</td>
<td style="text-align:right;">
0.85
</td>
</tr>
<tr>
<td style="text-align:left;">
minors
</td>
<td style="text-align:right;">
-0.05
</td>
<td style="text-align:right;">
0.26
</td>
<td style="text-align:right;">
-0.10
</td>
<td style="text-align:right;">
-0.21
</td>
<td style="text-align:right;">
0.15
</td>
<td style="text-align:right;">
0.22
</td>
<td style="text-align:right;">
0.50
</td>
<td style="text-align:right;">
0.71
</td>
<td style="text-align:right;">
0.61
</td>
</tr>
</tbody>
</table>

Reducing the dimensionality of a Term Document Matrix in this way with a "truncated" SVD of the first two columns of the three decomposed matricies may improve our ability to search documents when using cosine similarity (as we did before with the TF-IDF VSM). However, you should judge for yourself if this transformation gives better performance by comparing your search results with and without the trasnformation. The usefulness of will vary between corpora (collections of text documents).

Word context - going beyond count based word embeddings - Word2vec
==================================================================

TF-IDF VSM and LSA have been called [count based methods](http://clic.cimec.unitn.it/marco/publications/acl2014/baroni-etal-countpredict-acl2014.pdf). More recent methods use the context of words such as [word2vec](https://www.tensorflow.org/tutorials/representation/word2vec). Word embeddings like word2vec uses a context predicting approach that will be explained here intuitively soon too using this example from  [text2vec](http://text2vec.org/glove.html#word_embeddings).
