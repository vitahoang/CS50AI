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
    if page not in corpus:
        raise ValueError(f"Page '{page}' not found in corpus.")

    distribution = dict()
    total_pages = len(corpus)
    links = corpus[page]
    total_links = len(links)

    if total_links > 0:
        for p in corpus:
            distribution[p] = (1 - damping_factor) / total_pages
        for p in links:
            distribution[p] += damping_factor / total_links

    if total_links == 0:
        for p in corpus:
            distribution[p] = 1 / total_pages

    return distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    if n <= 0:
        raise ValueError("Number of samples 'n' must be a positive integer.")

    # Initialize count of visits for each page
    page_counts = {page: 0 for page in corpus}

    # Pick a starting page at random
    current_page = random.choice(list(corpus.keys()))
    page_counts[current_page] += 1

    # Repeat sampling n-1 times
    for _ in range(1, n):
        # Get the probability distribution for the next page
        distribution = transition_model(corpus, current_page, damping_factor)

        # Randomly pick the next page according to distribution
        pages, probs = zip(*distribution.items())
        current_page = random.choices(pages, weights=probs, k=1)[0]

        # Count the visit
        page_counts[current_page] += 1

    # Normalize counts to get probabilities
    page_ranks = {page: count / n for page, count in page_counts.items()}

    return page_ranks


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    iterate_pagerank = {page: 1 / len(corpus) for page in corpus}
    total_pages = len(corpus)
    for p in corpus:
        if len(corpus[p]) == 0:
            for page in corpus:
                corpus[p].add(page)

    converged = False
    while not converged:
        new_ranks = dict()
        for page in corpus:
            rank_sum = 0
            for linked_page in corpus:
                if page in corpus[linked_page]:
                    rank_sum += iterate_pagerank[linked_page] / \
                        len(corpus[linked_page])
            new_ranks[page] = (1 - damping_factor) / \
                total_pages + damping_factor * rank_sum

        converged = all(
            abs(new_ranks[page] - iterate_pagerank[page]) < 0.001 for page in corpus)
        iterate_pagerank = new_ranks
    return iterate_pagerank


if __name__ == "__main__":
    main()
