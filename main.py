#!/usr/bin/env python3

#-----------------------------------------------------------------------#
#
#    This program takes n-gram files and a word list    
#    and creates a file with lists of most similar words.
#    John Goldsmith and Wang Xiuli 2012.
#    Jackson Lee and Simon Jacobs 2015
#
#-----------------------------------------------------------------------#

from findManifold import *
import argparse


# argparse is a python module that makes creating command-line interfaces very very easy.
# Try: python main.py --help
def makeArgParser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("nWords", help="Number of words for analysis", type=int, default=9)
    parser.add_argument("nNeighbors", help="Number of neighbors", type=int, default=1000)
    parser.add_argument("nEigenvectors", help="Number of eigenvectors",
                        type=int, default=11)
    parser.add_argument("--bigrams", help="Bigrams file to use", type=str,
            default="../../data/english/ngrams/english-brown_bigrams.txt")
    parser.add_argument("--trigrams", help="Trigrams file to use", type=str,
            default="../../data/english/ngrams/english-brown_trigrams.txt")
    parser.add_argument("--words", help="Words file to use", type=str,
            default="../../data/english/ngrams/english-brown_words.txt")
    parser.add_argument("--output", help="Output folder to use", type=str,
            default="../../data/english/neighbors")
    parser.add_argument("--name", help="Corpus name", type=str, default="english-brown")
    parser.add_argument("--languagename", help="Language name", type=str, default="english")
    return parser

def DEBUG(s):
    print(s, flush=True)
#    sys.stdout.flush()

def main(argv):
    args = makeArgParser().parse_args()
    
    NumberOfWordsForAnalysis = args.nWords
    NumberOfNeighbors = args.nNeighbors
    NumberOfEigenvectors = args.nEigenvectors
    
    infileBigramsname = args.bigrams
    infileTrigramsname = args.trigrams
    infileWordsname = args.words
    wordfile = open(infileWordsname)
    trigramfile = open(infileTrigramsname)
    bigramfile = open(infileBigramsname)

    mywords = GetMyWords(wordfile)
    wordfile.close()
    print("Word file is ", infileWordsname, '\t corpus has ', len(mywords), ' words')

    if NumberOfWordsForAnalysis > len(mywords):
        NumberOfWordsForAnalysis = len(mywords)
        print('number of words for analysis reduced to ', NumberOfWordsForAnalysis)

    analyzedwordlist = list(mywords.keys())[ : NumberOfWordsForAnalysis] 

    outfilenameNeighbors = args.output + "/" + args.name + "_" + \
                           str(NumberOfWordsForAnalysis) + "_" + \
                           str(NumberOfNeighbors) + "_nearest_neighbors.txt"
    outfilenameContexts     = args.output + "/" + args.name + "_contexts.txt"
    outfilenameFromWordToContexts = args.output + "/" + args.name + "_" + \
                                    str(NumberOfWordsForAnalysis) + \
                                    "_from-word-to-contexts.txt"

    outfileFromWordToContexts = open(outfilenameFromWordToContexts, 'w')
    outfileNeighbors = open(outfilenameNeighbors, 'w')
    outfileContexts = open(outfilenameContexts, 'w')

    print("#  The number with each context is the number of distinct words found in that context.\n#", file=outfileContexts)

    for outfile in [outfileNeighbors, outfileFromWordToContexts]:
        print("# language: ", args.languagename,
              "\n# corpus: ", args.name,
              "\n#Number of words analyzed ", NumberOfWordsForAnalysis,
              "\n#Number of neighbors identified ", NumberOfNeighbors,"\n",
              file=outfile)

    print("\nI am looking for: ", infileTrigramsname)
    print("Number of words that will be analyzed: ", NumberOfWordsForAnalysis)
    print("Number of neighbors: ", NumberOfNeighbors)

    DEBUG("Reading bigrams/trigrams")

    # TODO: set two dicts -- (1) from word to context (2) from context to word
#    context_array, wordContextDict, contextWordDict = GetContextArrayNew(analyzedwordlist, bigramfile, trigramfile)
#    context_array = GetContextArrayNew(wordContextDict, contextWordDict)

    DEBUG("Computing context array")
    context_array = GetContextArray(NumberOfWordsForAnalysis,
                                    mywords, bigramfile, trigramfile)

    DEBUG("Computing shared contexts")
    CountOfSharedContexts = context_array.dot(context_array.T).todense()

    DEBUG("Computing diameter")
    Diameter = Normalize(NumberOfWordsForAnalysis, CountOfSharedContexts)

    DEBUG("Computing incidence graph")
    incidencegraph = compute_incidence_graph(NumberOfWordsForAnalysis, Diameter, CountOfSharedContexts)
    
    DEBUG("Computing mylaplacian")
    mylaplacian = compute_laplacian(NumberOfWordsForAnalysis, Diameter, incidencegraph)

    DEBUG("Compute eigenvectors...")
    myeigenvalues, myeigenvectors = GetEigenvectors(mylaplacian)
   

    DEBUG('Coordinates computed. now computing distances between words...')
    coordinates = myeigenvectors[:,:NumberOfEigenvectors] # take first N columns of eigenvector matrix
    wordsdistance = compute_words_distance(NumberOfWordsForAnalysis, coordinates)

    DEBUG('Computing nearest neighbors now... ')
    closestNeighbors = compute_closest_neighbors(analyzedwordlist, wordsdistance, NumberOfNeighbors)

    DEBUG("Output to files")
    for (wordno, word) in enumerate(analyzedwordlist):
        print(' '.join([analyzedwordlist[idx] for idx in closestNeighbors[wordno]]), file=outfileNeighbors)

    outfileNeighbors.close()



# Don't execute any code if we are loading this file as a module.
if __name__ == "__main__":
    main(sys.argv)
