import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():

    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])

    #corpus = crawl('corpus0')

    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """

    totalPages = corpus.copy()
    possibleLinks = corpus[page]

    for page in totalPages:
        totalPages[page] = (1 - damping_factor) / len(corpus)
        if page in possibleLinks and len(possibleLinks) > 0:
            totalPages[page] += 0.85 / len(possibleLinks)

    return totalPages

    raise NotImplementedError


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    pageRank = corpus.fromkeys(corpus, 0)

    page = random.choice(list(corpus))

    for i in range(n):
        pageChoosingProb = transition_model(corpus, page, damping_factor)
        page = random.choices(list(corpus.keys()), weights=list(pageChoosingProb.values()), k=1)[0]
        pageRank[page] += 1/n
    
    return pageRank

    raise NotImplementedError


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    currentPageRank = corpus.fromkeys(corpus, 0)
    previousIterationPageRank = corpus.fromkeys(corpus, 1/len(corpus))
    pointers = corpus.fromkeys(corpus, None)

    for page in corpus:
        pagePoint = set()
        for pageLink in corpus: 
            if page in corpus[pageLink]:
                pagePoint.add(pageLink)
        pointers[page] = pagePoint


    iterations = 0
    isNotAccurate = True

    while isNotAccurate and iterations < 100000: 
        for page in currentPageRank:
            sum = 0
            for pointer in pointers[page]:
                sum += previousIterationPageRank[pointer] / len(corpus[pointer])
            currentPageRank[page] = (1 - damping_factor)/len(corpus) + damping_factor * sum
        
        isNotAccurate = notAccurate(previousIterationPageRank, currentPageRank)
        previousIterationPageRank = currentPageRank.copy()

        iterations += 1

    print("Total Number of Iterations: ", iterations)
    return currentPageRank
    
    raise NotImplementedError

def notAccurate(prev, current):
    
    for page in current:
        if abs(prev[page] - current[page]) > 0.001:
            return True
    
    return False


if __name__ == "__main__":
    main()
