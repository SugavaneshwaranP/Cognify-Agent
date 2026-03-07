"""
CognifyX – Candidate Ranker
Composite ranking that combines ATS, keyword, and LLM scores.
"""
import re
import json


class CandidateRanker:
    def __init__(self, agent):
        self.agent = agent

    def rank(self, jd, profile, identifier="unknown"):
        """Call LLM for scoring."""
        return self.agent.score_candidate(jd, profile, identifier=identifier)

    @staticmethod
    def extract_llm_score(analysis_text):
        """
        Extract numerical score from LLM analysis text.
        Looks for patterns like 'Score: 75/100', '85 out of 100', 'score of 80', etc.
        """
        if not analysis_text:
            return 50  # Default mid-score

        text = str(analysis_text)

        # Pattern 1: "Score: 75/100"
        match = re.search(r'[Ss]core[:\s]*(\d{1,3})\s*/\s*100', text)
        if match:
            return min(100, int(match.group(1)))

        # Pattern 2: "Heuristic Analysis Score: 75"
        match = re.search(r'[Ss]core[:\s]*(\d{1,3})', text)
        if match:
            return min(100, int(match.group(1)))

        # Pattern 3: "85 out of 100"
        match = re.search(r'(\d{1,3})\s*(?:out of|\/)\s*100', text)
        if match:
            return min(100, int(match.group(1)))

        # Pattern 4: any standalone number 0-100
        match = re.search(r'\b(\d{2,3})\b', text)
        if match:
            val = int(match.group(1))
            if 0 <= val <= 100:
                return val

        return 50

    @staticmethod
    def compute_composite_score(ats_score, keyword_score, llm_score):
        """
        Weighted composite score:
        - ATS (TF-IDF similarity): 25%
        - Keyword match: 35%
        - LLM analysis score: 40%
        """
        composite = (
            ats_score * 0.25 +
            keyword_score * 0.35 +
            llm_score * 0.40
        )
        return round(min(100, max(0, composite)), 2)

    @staticmethod
    def rank_candidates(profiles):
        """
        Assign final ranks based on composite score.
        Returns sorted list with final_rank assigned.
        """
        # Sort by composite score descending
        sorted_profiles = sorted(profiles, key=lambda x: x.get('composite_score', 0), reverse=True)

        for i, p in enumerate(sorted_profiles):
            p['final_rank'] = i + 1

        return sorted_profiles
