"""
CognifyX – LLM Extractor
Thin wrapper around LLMAgent for structured data extraction.
"""


class LLMExtractor:
    def __init__(self, agent):
        self.agent = agent

    def extract(self, text, identifier="unknown"):
        return self.agent.extract_structured_data(text, identifier=identifier)
