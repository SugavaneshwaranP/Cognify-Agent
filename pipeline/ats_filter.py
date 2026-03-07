"""
CognifyX – ATS Filter
Hybrid scoring using user-provided keywords + TF-IDF cosine similarity.
Keyword matching is the PRIMARY scoring dimension based on user input.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class ATSFilter:
    def __init__(self, top_n=40):
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer(stop_words='english')

    def _parse_keywords(self, keyword_input):
        """
        Parse user keyword input. Accepts comma-separated or newline-separated keywords.
        Returns a list of lowercase keyword phrases.
        """
        if not keyword_input:
            return []
        # Split by comma or newline
        raw = re.split(r'[,\n]+', keyword_input)
        keywords = [k.strip().lower() for k in raw if k.strip()]
        return keywords

    def _keyword_match_score(self, resume_text, keywords):
        """
        Calculate keyword match score (0-100).
        Supports exact phrase match and individual word match.
        """
        if not keywords:
            return 0, [], []

        text_lower = resume_text.lower()
        matched = []
        missed = []

        for kw in keywords:
            # Exact phrase match (e.g., "machine learning")
            if kw in text_lower:
                matched.append(kw)
            else:
                # Partial word match – check if all words in multi-word keyword exist
                words = kw.split()
                if len(words) > 1 and all(w in text_lower for w in words):
                    matched.append(kw)
                else:
                    missed.append(kw)

        if not keywords:
            return 0, matched, missed

        score = (len(matched) / len(keywords)) * 100
        return round(score, 2), matched, missed

    def calculate_scores(self, resumes, job_description, keywords_input=""):
        """
        Calculates ATS scores using:
        - 60% weight: User keyword matching
        - 40% weight: TF-IDF Cosine Similarity with JD
        
        Args:
            resumes: list of dicts with 'text', 'filename', 'path'
            job_description: the full JD text
            keywords_input: comma/newline separated keywords from user
        
        Returns:
            Sorted list of top_n resumes with scores and keyword match info.
        """
        if not resumes:
            return []

        keywords = self._parse_keywords(keywords_input)

        # --- TF-IDF Cosine Similarity ---
        texts = [r['text'] for r in resumes]
        all_texts = [job_description] + texts

        try:
            tfidf_matrix = self.vectorizer.fit_transform(all_texts)
            cosine_sims = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()
        except Exception:
            cosine_sims = [0.0] * len(resumes)

        # --- Score each resume ---
        for i, resume in enumerate(resumes):
            # Keyword score (0-100)
            kw_score, matched, missed = self._keyword_match_score(resume['text'], keywords)

            # TF-IDF similarity score (0-100)
            tfidf_score = cosine_sims[i] * 100

            # Combined ATS Score
            if keywords:
                # When keywords provided: 60% keyword + 40% TF-IDF
                combined = (kw_score * 0.6) + (tfidf_score * 0.4)
            else:
                # No keywords: 100% TF-IDF
                combined = tfidf_score

            # Floor at 2.0 to avoid zero for valid resumes
            resume['score'] = round(max(2.0, min(100.0, combined)), 2)
            resume['keyword_score'] = kw_score
            resume['keywords_matched'] = matched
            resume['keywords_missed'] = missed
            resume['tfidf_score'] = round(tfidf_score, 2)

        # Sort and return top N
        ranked = sorted(resumes, key=lambda x: x['score'], reverse=True)
        return ranked[:self.top_n]
