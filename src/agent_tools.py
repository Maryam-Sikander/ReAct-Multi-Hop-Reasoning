import re
import requests

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

WIKI_API = "https://en.wikipedia.org/w/api.php"


class WikiEnv:
    def __init__(self, intro_sentences=5):
        self.intro_sentences = intro_sentences
        self._sentences = []
        self._lookup_keyword = None
        self._lookup_pos = 0
        self.headers = {"User-Agent": "ReActBaselineReplication/1.0 (personal research project)"}
        self._search_cache = {}

        

    def search(self, entity):
        if entity in self._search_cache:
            cached = self._search_cache[entity]
            self._sentences = cached["sentences"]
            self._lookup_keyword = None
            self._lookup_pos = 0
            return cached["intro"]

        page = self._fetch_extract(entity)
        if page is None:
            similar = self._fetch_suggestions(entity)
            self._sentences = []
            self._lookup_keyword = None
            result = f"Could not find [{entity}]. Similar: {similar}" if similar else f"Could not find [{entity}]."
            self._search_cache[entity] = {"sentences": [], "intro": result}
            return result

        title, extract = page
        self._sentences = [s for s in SENTENCE_SPLIT.split(extract) if s.strip()]
        self._lookup_keyword = None
        self._lookup_pos = 0
        intro = " ".join(self._sentences[: self.intro_sentences]) or f"[{title}] found but has no readable text."
        self._search_cache[entity] = {"sentences": self._sentences, "intro": intro}
        return intro

    def lookup(self, keyword):
        if not self._sentences:
            return "No page open. Use Search first."
        if keyword != self._lookup_keyword:
            self._lookup_keyword = keyword
            self._lookup_pos = 0
        matches = [i for i, s in enumerate(self._sentences) if keyword.lower() in s.lower()]
        if not matches:
            return f"No mentions of [{keyword}] found."
        remaining = [i for i in matches if i >= self._lookup_pos]
        if not remaining:
            return f"No more results for [{keyword}]."
        idx = remaining[0]
        self._lookup_pos = idx + 1
        result_num = matches.index(idx) + 1
        return f"(Result {result_num}/{len(matches)}) {self._sentences[idx]}"

    def _fetch_extract(self, title):
        params = {"action": "query", "prop": "extracts", "explaintext": 1,
                  "redirects": 1, "titles": title, "format": "json"}
        resp = requests.get(WIKI_API, params=params, headers=self.headers, timeout=15)
        resp.raise_for_status()
        pages = resp.json().get("query", {}).get("pages", {})
        for page_id, page in pages.items():
            if page_id == "-1" or "missing" in page:
                return None
            extract = page.get("extract", "")
            if extract.strip():
                return page.get("title", title), extract
        return None

    def _fetch_suggestions(self, query, limit=5):
        params = {"action": "query", "list": "search", "srsearch": query,
                  "srlimit": limit, "format": "json"}
        resp = requests.get(WIKI_API, params=params, headers=self.headers, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("query", {}).get("search", [])
        return ", ".join(r["title"] for r in results)


    