#!/usr/bin/env python3

#-----------------------------------------------------------------------#
#
#    This program takes n-gram files and a word list    
#    and creates a file with lists of most similar words.
#    John Goldsmith and Wang Xiuli 2012.
#    Jackson Lee and Simon Jacobs 2015
#
#-----------------------------------------------------------------------#

import sys
import math
import collections
import numpy as np
import itertools
import time

#import multiprocessing as mp

import scipy.spatial.distance as sd
import scipy.sparse as sp
import scipy.sparse.linalg as sl

def Normalize(NumberOfWordsForAnalysis, CountOfSharedContexts):
    arr = np.ones((NumberOfWordsForAnalysis))
    for w in range(NumberOfWordsForAnalysis):
        arr[w] = np.sum(CountOfSharedContexts[w]) - CountOfSharedContexts[w, w]
    return arr

def GetMyWords(wordfile):
    mywords = collections.OrderedDict()
    for line in wordfile:
        pieces = line.split()
        if pieces[0] == "#":
            continue
        mywords[pieces[0]] = int(pieces[1])         
    return mywords

def GetWordContextDicts(wordlist, bigramfile, trigramfile):
    wordContextDict = collections.OrderedDict()
    contextWordDict = collections.defaultdict(set)

    def addwordcontext(word, context):
        wordContextDict[word].add(context)
        contextWordDict[context].add(word)

    for word in wordlist:
        wordContextDict[word] = set()

    for line in trigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if c[0] in wordlist:
            addwordcontext(c[0], "__" + c[1] + c[2])
        if c[1] in wordlist:
            addwordcontext(c[1], c[0] + "__" + c[2])
        if c[2] in wordlist:
            addwordcontext(c[2], c[0] + c[1] + "__")

    for line in bigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if c[0] in wordlist:
            addwordcontext(c[0], "__" + c[1])
        if c[1] in wordlist:
            addwordcontext(c[1], c[0] + "__")

    return (wordContextDict, contextWordDict)



def GetContextArray(nwords, mywords, bigramfile, trigramfile):

    wordlist = list(mywords.keys())[ : nwords]
    
    class Namespace:
        pass
    ns = Namespace() # this is necessary so we can reference ncontexts from inner functions
    ns.ncontexts = 0
    def contexts_incr():
        tmp = ns.ncontexts
        ns.ncontexts += 1
        return tmp
    contextdict = collections.defaultdict(contexts_incr)
    worddict = {w: wordlist.index(w) for w in wordlist}

    # entries for sparse matrix
    rows = []
    cols = []
    vals = [] 

    def addword(word, context):
        w = worddict[word]
        c = contextdict[context]
        rows.append(w)
        cols.append(c)
        vals.append(1)

    for line in trigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if worddict.get(c[0]) is not None:
            addword(c[0], "__" + c[1] + c[2])
        if worddict.get(c[1]) is not None:
            addword(c[1], c[0] + "__" + c[2])
        if worddict.get(c[2]) is not None:
            addword(c[2], c[0] + c[1] + "__")

    for line in bigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if worddict.get(c[0]) is not None:
            addword(c[0], "__" + c[1])
        if worddict.get(c[1]) is not None:
            addword(c[1], c[0] + "__")
    
    return sp.csr_matrix((vals,(rows,cols)), shape=(nwords, ns.ncontexts) )


# TODO: need this function to keep track of (i.e., generate and create) wordContextDict, contextWordDict?
#       The problem with the function is that it is slow as it stands now...
def GetContextArrayNew(wordlist, bigramfile, trigramfile):
    wordContextDict = collections.defaultdict(set)
    contextWordDict = collections.defaultdict(set)

    nwords = len(wordlist)
#    wordlist = list(mywords.keys())[ : nwords]
    
    class Namespace:
        pass
    ns = Namespace() # this is necessary so we can reference ncontexts from inner functions
    ns.ncontexts = 0
    def contexts_incr():
        tmp = ns.ncontexts
        ns.ncontexts += 1
        return tmp
    contextdict = collections.defaultdict(contexts_incr)
    worddict = {w: wordlist.index(w) for w in wordlist}

    # entries for sparse matrix
    rows = []
    cols = []
    vals = [] 

    def addword(word, context):
        w = worddict[word]
        c = contextdict[context]
        rows.append(w)
        cols.append(c)
        vals.append(1)
        wordContextDict[word].add(context)
        contextWordDict[context].add(word)

    for line in trigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if worddict.get(c[0]) is not None:
            addword(c[0], ' '.join(["_", c[1], c[2]]))
        if worddict.get(c[1]) is not None:
            addword(c[1], ' '.join([c[0], "_", c[2]]))
        if worddict.get(c[2]) is not None:
            addword(c[2], ' '.join([c[0], c[1], "_"]))

    for line in bigramfile:
        if line.startswith('#'):
            continue
        c = line.split()
        if worddict.get(c[0]) is not None:
            addword(c[0], "_ " + c[1])
        if worddict.get(c[1]) is not None:
            addword(c[1], c[0] + " _")

    print('len(contextWordDict) ', len(contextWordDict))

    wordlistForContextCheck = wordlist[:100]
    nContextsWithMostFreqWords = 0
    for context in contextWordDict.keys():
        contextIndividualWordsList = context.split()

        for word in contextIndividualWordsList:
            if word in wordlistForContextCheck:
                nContextsWithMostFreqWords += 1
                break

    print('nContextsWithMostFreqWords ', nContextsWithMostFreqWords)

    return (sp.csr_matrix((vals,(rows,cols)), shape=(nwords, ns.ncontexts) ),
            wordContextDict, contextWordDict)



def counting_context_features(context_array):
    return np.dot(context_array, context_array.T) 


def compute_incidence_graph(NumberOfWordsForAnalysis, Diameter, CountOfSharedContexts):
    incidencegraph= np.asarray(CountOfSharedContexts, dtype=np.int32)

    for w in range(NumberOfWordsForAnalysis):
        incidencegraph[w, w] = Diameter[w]
    return incidencegraph



def compute_laplacian(NumberOfWordsForAnalysis, Diameter, incidencegraph): 
    D = np.sqrt(np.outer(Diameter, Diameter))
    D[D==0] = 1 # we want to NOT have div-by-zero errors, but if D[i,j] = 0 then incidencegraph[i,j] = 0 too.
    mylaplacian = (1/D) * incidencegraph # broadcasts the multiplication, so A[i,j] = B[i,j] * C[i, j]
    return mylaplacian

def compute_coordinates(NumberOfWordsForAnalysis, NumberOfEigenvectors, myeigenvectors):
    Coordinates = dict()
    for wordno in range(NumberOfWordsForAnalysis):
        Coordinates[wordno]= list() 
        for eigenno in range(NumberOfEigenvectors):
            Coordinates[wordno].append( myeigenvectors[ wordno, eigenno ] )
    return Coordinates



def compute_words_distance(nwords, coordinates):
    #def distance(u, v):
    #    return np.sum(np.abs(np.power(u - v, 3)));
    return sd.squareform(sd.pdist(coordinates, "euclidean"))


def compute_closest_neighbors(analyzedwordlist, wordsdistance, NumberOfNeighbors):
    sortedNeighbors = wordsdistance.argsort() # indices of sorted rows, low to high
    closestNeighbors = sortedNeighbors[:,:NumberOfNeighbors+1] # truncate columns at NumberOfNeighbors+1 
    return closestNeighbors


def GetEigenvectors(laplacian):
    laplacian_sparse = sp.csr_matrix(laplacian)
    return sl.eigs(laplacian_sparse)


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#


# not used
def ReadInTrigrams(trigramfile, analyzedwordlist, analyzedwordset, from_word_to_context):
    for line in trigramfile:
        if line.startswith('#'):
            continue
        thesewords = line.split()

        thisword = thesewords[1]
        if thisword in analyzedwordset:
            context = thesewords[0] + " __ " +  thesewords[2]
            wordno = analyzedwordlist.index(thisword)
            from_word_to_context[wordno][context] += 1

        #Left trigrams
        thisword = thesewords[0]
        if thisword in analyzedwordset:
            context = " __ " + thesewords[1] + " " + thesewords[2]
            wordno = analyzedwordlist.index(thisword)
            from_word_to_context[wordno][context] += 1

        #Right trigrams
        thisword = thesewords[2]
        if thisword   in analyzedwordset:    
            context = thesewords[0] + " " + thesewords[1] + " __ "
            wordno = analyzedwordlist.index(thisword)
            from_word_to_context[wordno][context] += 1

# not used
def ReadInBigrams(bigramfile, analyzedwordlist, analyzedwordset, from_word_to_context):
    print("...Reading in bigram file.")
    for line in bigramfile:
        thesewords = line.split()
        if thesewords[0] == "#":
            continue 
        thisword = thesewords[1]
        if thisword in analyzedwordset:
            context = thesewords[0] + " __ " 
            wordno = analyzedwordlist.index(thisword) # TODO: is this a bug in the older version?
            from_word_to_context[wordno][context] += 1
        thisword = thesewords[0]
        if thisword in analyzedwordset:
            context = "__ " + thesewords[1]
            wordno = analyzedwordlist.index(thisword)
            from_word_to_context[wordno][context] += 1


# not used
def MakeContextArray(NumberOfWordsForAnalysis, from_word_to_context):
    context_list = list(set(context for i in from_word_to_context for context in from_word_to_context[i]))
    context_array = np.zeros((NumberOfWordsForAnalysis, len(context_list)))
    for wordno in range(NumberOfWordsForAnalysis):
        for contextno in range(len(context_list)):
            if (from_word_to_context[wordno][(context_list[contextno])] > 0):
                context_array[wordno, contextno] = 1
    return context_array

# not used
def QuickGetNumberOfSharedContexts(word1, word2):
    return np.dot(context_array[word1], context_array[word2])

# not used
def GetNumberOfSharedContexts(word1, word2, from_word_to_context):
    return len(set(from_word_to_context[word1]) & set(from_word_to_context[word2]))

# not used
def counting_context_features_old(NumberOfWordsForAnlysis, from_word_to_context):
    arr = np.zeros((NumberOfWordsForAnalysis, NumberOfWordsForAnalysis))
    for word1 in range(0, NumberOfWordsForAnalysis):
        for word2 in range(word1+1, NumberOfWordsForAnalysis):
            x = GetNumberOfSharedContexts(word1, word2, from_word_to_context)
            arr[word1, word2] = x
            arr[word2, word1] = x
    return arr


# not used
def compute_incidence_graph_old(NumberOfWordsForAnalysis, Diameter, CountOfSharedContexts):

    for (w1, w2) in itertools.product(range(NumberOfWordsForAnalysis), repeat=2):
        if w1 == w2:
            incidencegraph[w1,w1] = Diameter[w1]
        else:
            incidencegraph[w1,w2] = CountOfSharedContexts[w1,w2]    

    return incidencegraph

# not used
def compute_laplacian_old(NumberOfWordsForAnalysis, Diameter, incidencegraph):
    mylaplacian = np.zeros((NumberOfWordsForAnalysis, NumberOfWordsForAnalysis), dtype=np.float32 )

    for (i, j) in itertools.product(range(NumberOfWordsForAnalysis), repeat=2):
        if i == j:
            mylaplacian[i,j] = 1
        else:
            if incidencegraph[i,j] == 0:
                mylaplacian[i,j]=0
            else:
                mylaplacian[i,j] = -1 * incidencegraph[i,j]/ math.sqrt ( Diameter[i] * Diameter[j] )
    return mylaplacian


# not used
def compute_words_distance_old(nwords, coordinates):
    arr = np.zeros((nwords, nwords))
    for wordno1 in range(nwords):
        for wordno2 in range(wordno1+1, nwords):
            distance = 0
            x = coordinates[wordno1] - coordinates[wordno2]
            distance = np.sum(np.abs(np.power(x, 3)))
            arr[wordno1, wordno2] = distance
            arr[wordno2, wordno1] = distance
    
    return arr


# not used
def compute_closest_neighbors_old(analyzedwordlist, wordsdistance, NumberOfNeighbors):
    closestNeighbors = dict()
    for (wordno1, word1) in enumerate(analyzedwordlist):
        neighborWordNumberList = [wordno2 for (wordno2, distance) in sorted(enumerate(list(wordsdistance[wordno1])), key=lambda x:x[1])][1:]
        neighborWordNumberList = neighborWordNumberList[: NumberOfNeighbors+1]
        closestNeighbors[wordno1] = neighborWordNumberList
    return closestNeighbors
    


#------------------------------------------------------------------------------#
#------------------------------------------------------------------------------#
# TODO: bring latex output back


#def LatexAndEigenvectorOutput(LatexFlag, PrintEigenvectorsFlag, infileWordsname, outfileLatex, outfileEigenvectors, NumberOfEigenvectors, myeigenvalues, NumberOfWordsForAnalysis):
#    if LatexFlag:
#            #Latex output
#            print("% ",  infileWordsname, file=outfileLatex)
#            print("\\documentclass{article}", file=outfileLatex) 
#            print("\\usepackage{booktabs}" , file=outfileLatex)
#            print("\\begin{document}" , file=outfileLatex)

#    data = dict() # key is eigennumber, value is list of triples: (index, word, eigen^{th} coordinate) sorted by increasing coordinate
#    print("9. Printing contexts to latex file.")
#    formatstr = '%20s   %15s %10.3f'
#    headerformatstr = '%20s  %15s %10.3f %10s'
#    NumberOfWordsToDisplayForEachEigenvector = 20
#            

#            
#                     
#    if PrintEigenvectorsFlag:

#            for eigenno in range(NumberOfEigenvectors):
#                    print >>outfileEigenvectors
#                    print >>outfileEigenvectors,headerformatstr %("Eigenvector number", eigenno, myeigenvalues[eigenno], "word" )
#                    print >>outfileEigenvectors,"_"*50 

#                    eigenlist=list()		
#                    for wordno in range (NumberOfWordsForAnalysis):		 
#                            eigenlist.append( (wordno,myeigenvectors[wordno, eigenno]) )			
#                    eigenlist.sort(key=lambda x:x[1])			

#                    for wordno in range(NumberOfWordsForAnalysis):	
#                            word = analyzedwordlist[eigenlist[wordno][0]]
#                            coord =  eigenlist[wordno][1]		
#                            print >>outfileEigenvectors, formatstr %(eigenno, word, eigenlist[wordno][1])


#     

#    if LatexFlag:
#            for eigenno in range(NumberOfEigenvectors):
#                    eigenlist=list()	
#                    data = list()
#                    for wordno in range (NumberOfWordsForAnalysis):		 
#                            eigenlist.append( (wordno,myeigenvectors[wordno, eigenno]) )			
#                    eigenlist.sort(key=lambda x:x[1])			
#                    print >>outfileLatex			 
#                    print >>outfileLatex, "Eigenvector number", eigenno, "\n" 
#                    print >>outfileLatex, "\\begin{tabular}{lll}\\toprule"
#                    print >>outfileLatex, " & word & coordinate \\\\ \\midrule "

#                    for i in range(NumberOfWordsForAnalysis):			 
#                            word = analyzedwordlist[eigenlist[i][0]]
#                            coord =  eigenlist[i][1]
#                            if i < NumberOfWordsToDisplayForEachEigenvector or i > NumberOfWordsForAnalysis - NumberOfWordsToDisplayForEachEigenvector:
#                                    data.append((i, word , coord ))
#                    for (i, word, coord) in data:
#                            if word == "&":
#                                    word = "\&" 
#                            print >>outfileLatex,  "%5d & %10s &  %10.3f \\\\" % (i, word, coord) 

#                    print >>outfileLatex, "\\bottomrule \n \\end{tabular}", "\n\n"
#                    print >>outfileLatex, "\\newpage" 
#            print >>outfileLatex, "\\end{document}" 


