# find-manifold

Use bigrams and trigrams of a given language to explore word distribution

## Input and output files

Input: n-gram files, like those generated from [maketrigrams](https://github.com/JohnAGoldsmith/maketrigrams)

Output: a dict where a key is a word (call it "keyword") and the value is a list of words that are distributionally most similar to the keyword

## Scripts

``main.py`` loads ``findManifold.py`` as a module. Run with Python 3, not Python 2.

## Authors

John Goldsmith and Wang Xiuli, 2012

Jackson Lee and Simon Jacobs, 2015


