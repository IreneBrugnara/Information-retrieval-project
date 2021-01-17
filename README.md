# Information retrieval project

This is the final project for the course in Information Retrieval at DSSC.

The code is written in Python3, version 3.8.5.
The file "ProbabilisticModel.py" contains the classes of the IR system and functions for reading the dataset. The jupiter notebook "UsageExamples.ipynb" explains how to use the IR system. The file "Evaluation.ipynb" tests the effectiveness of the IR system.

The code for building the inverted index is that seen during the course lectures (with the only difference that I perform binary search instead of linear scan to search in the inverted index). I wrote the code for answering queries in a probabilistic framework. The user can perform relevance feedback and also pseudo-relevance feedback.

I used the "LISA collection" as dataset for testing my information retrieval system. It contains titles and abstracts of 6004 articles (actually 5999 after correcting errors in the files) from the Library and Information Science Abstracts database. The dataset can be downloaded here:
http://ir.dcs.gla.ac.uk/resources/test_collections/lisa/

