"""
CognifyX – Final Analysis
Generates executive-level insights from top candidates.
"""


class FinalAnalysis:
    def __init__(self, agent):
        self.agent = agent

    def analyze(self, top_candidates_summary):
        return self.agent.final_intelligence(top_candidates_summary)
