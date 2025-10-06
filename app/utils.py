import wikipedia

def search_wikipedia(query, max_results=5):
    try:
        results = wikipedia.search(query, results=max_results)
        pages = []
        for title in results:
            try:
                summary = wikipedia.summary(title, sentences=3)
                pages.append(f"{title}: {summary}")
            except Exception:
                continue
        return pages
    except Exception:
        return []
