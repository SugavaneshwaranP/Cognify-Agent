"""
CognifyX – Conversational Chat Agent
Handles interactive human-in-the-loop conversations about pipeline results.
Supports: asking context-aware questions, re-ranking based on feedback,
explaining reasoning, comparing candidates, and refining criteria.
"""
import json
import re
from pipeline.candidate_ranker import CandidateRanker


class ChatAgent:
    """
    Interactive conversational agent for post-pipeline Q&A and refinement.
    Works with or without LLM (Ollama) – uses smart heuristic fallback.
    """

    # Intent patterns for classifying user messages
    INTENT_PATTERNS = {
        "compare": [
            r"compare\b", r"difference between", r"vs\.?", r"versus",
            r"who is better", r"which one", r"between .+ and"
        ],
        "explain": [
            r"explain\b", r"why\b", r"how come", r"reasoning",
            r"tell me more", r"details? about", r"elaborate"
        ],
        "rerank": [
            r"re-?rank", r"prioritize", r"prefer", r"weight",
            r"importance", r"focus on", r"value .+ more",
            r"remote", r"budget", r"salary", r"degree",
            r"experience .+ important", r"skills? .+ important",
            r"education"
        ],
        "filter": [
            r"remove\b", r"exclude\b", r"filter out", r"only .+ with",
            r"minimum\b", r"at least", r"must have", r"require"
        ],
        "summary": [
            r"summarize", r"summary", r"overview", r"brief",
            r"top pick", r"recommendation", r"who should"
        ],
        "candidate_query": [
            r"(?:tell me|what) about (.+)", r"candidate #?\d+",
            r"rank #?\d+", r"number \d+"
        ],
        "greeting": [
            r"^(hi|hello|hey|good morning|good evening)\b",
            r"how are you", r"what can you do"
        ]
    }

    def __init__(self, llm_agent=None):
        self.llm_agent = llm_agent
        self.conversation_history = []
        self.preferences = {}  # Accumulated user preferences

    def _classify_intent(self, message):
        """Classify the user's message into an intent category."""
        msg_lower = message.lower().strip()

        for intent, patterns in self.INTENT_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, msg_lower, re.IGNORECASE):
                    return intent

        return "general"

    def _find_candidate_by_name(self, name_query, candidates):
        """Find a candidate by partial name match."""
        name_query = name_query.lower().strip()
        for c in candidates:
            candidate_name = c.get('name', c.get('filename', '')).lower()
            if name_query in candidate_name or candidate_name in name_query:
                return c

        # Try matching by rank number
        rank_match = re.search(r'#?(\d+)', name_query)
        if rank_match:
            rank = int(rank_match.group(1))
            for c in candidates:
                if c.get('final_rank') == rank:
                    return c
        return None

    def _format_candidate_summary(self, c):
        """Format a single candidate into a readable summary."""
        skills_text = ", ".join(c.get('skills', [])[:8]) or "Not listed"
        matched = ", ".join(c.get('keywords_matched', [])[:5]) or "None"
        missed = ", ".join(c.get('keywords_missed', [])[:5]) or "None"

        return (
            f"**#{c.get('final_rank', '?')} {c.get('name', c.get('filename', 'Unknown'))}**\n"
            f"- 📊 Composite Score: **{c.get('composite_score', 0)}** "
            f"(ATS: {c.get('ats_score', 0)} | Keywords: {c.get('keyword_score', 0)}% | "
            f"LLM: {c.get('llm_score', 0)})\n"
            f"- 🏷️ Domain: {c.get('domain', 'General')}\n"
            f"- 💼 Experience: {c.get('experience_years', '?')} years\n"
            f"- 🎓 Education: {c.get('education', 'Not specified')}\n"
            f"- 🔧 Skills: {skills_text}\n"
            f"- ✅ Keywords Matched: {matched}\n"
            f"- ❌ Keywords Missed: {missed}"
        )

    def _handle_greeting(self, message, results):
        """Handle greeting messages."""
        if not results:
            return (
                "👋 Hello! I'm the **CognifyX AI Assistant**. I can help you analyze "
                "candidates once the pipeline has run.\n\n"
                "Upload resumes and run the screening pipeline first, then come chat with me!"
            )

        total = results.get('total_parsed', 0)
        shortlisted = len(results.get('shortlisted', []))
        top_name = ""
        if results.get('shortlisted'):
            top_name = results['shortlisted'][0].get('name', 'Unknown')

        return (
            f"👋 Hello! I'm the **CognifyX AI Assistant**.\n\n"
            f"I've analyzed **{total}** resumes and shortlisted **{shortlisted}** candidates. "
            f"The top candidate is **{top_name}**.\n\n"
            f"Here's what you can ask me:\n"
            f"- 🔍 **\"Compare candidate #1 and #3\"** — Side-by-side comparison\n"
            f"- 💬 **\"Why is #2 ranked higher than #4?\"** — Explain reasoning\n"
            f"- 🎯 **\"Prioritize experience over skills\"** — Re-rank with your preferences\n"
            f"- 🚫 **\"Remove candidates with less than 3 years\"** — Filter results\n"
            f"- 📝 **\"Give me a summary\"** — Quick overview\n"
            f"- 👤 **\"Tell me about {top_name}\"** — Deep dive into a candidate"
        )

    def _handle_compare(self, message, results):
        """Handle candidate comparison requests."""
        candidates = results.get('shortlisted', results.get('candidates', []))
        if not candidates:
            return "❌ No candidates available to compare. Please run the pipeline first."

        # Try to extract two candidate references
        numbers = re.findall(r'#?(\d+)', message)
        names = re.findall(r'(?:compare|between)\s+(.+?)\s+(?:and|vs|with)\s+(.+)', message, re.I)

        c1, c2 = None, None

        if len(numbers) >= 2:
            for c in candidates:
                if c.get('final_rank') == int(numbers[0]):
                    c1 = c
                if c.get('final_rank') == int(numbers[1]):
                    c2 = c
        elif names:
            c1 = self._find_candidate_by_name(names[0][0], candidates)
            c2 = self._find_candidate_by_name(names[0][1], candidates)

        if not c1 or not c2:
            # Default to top 2
            if len(candidates) >= 2:
                c1, c2 = candidates[0], candidates[1]
            else:
                return "I need at least 2 candidates to compare. Try: **\"Compare #1 and #2\"**"

        # Build comparison
        n1 = c1.get('name', c1.get('filename', 'Candidate 1'))
        n2 = c2.get('name', c2.get('filename', 'Candidate 2'))

        comparison = f"## 🔍 Comparison: {n1} vs {n2}\n\n"
        comparison += "| Metric | " + n1 + " | " + n2 + " | Edge |\n"
        comparison += "|---|---|---|---|\n"

        metrics = [
            ("Composite Score", 'composite_score', True),
            ("ATS Score", 'ats_score', True),
            ("Keyword Match", 'keyword_score', True),
            ("LLM Score", 'llm_score', True),
            ("Experience", 'experience_years', True),
        ]

        for label, key, higher_better in metrics:
            v1 = c1.get(key, 0) or 0
            v2 = c2.get(key, 0) or 0
            suffix = "%" if key == 'keyword_score' else (" yrs" if key == 'experience_years' else "")

            if v1 > v2:
                edge = f"✅ {n1}" if higher_better else f"✅ {n2}"
            elif v2 > v1:
                edge = f"✅ {n2}" if higher_better else f"✅ {n1}"
            else:
                edge = "🤝 Tie"

            comparison += f"| {label} | {v1}{suffix} | {v2}{suffix} | {edge} |\n"

        # Skills comparison
        s1 = set(s.lower() for s in c1.get('skills', []))
        s2 = set(s.lower() for s in c2.get('skills', []))
        common = s1 & s2
        only1 = s1 - s2
        only2 = s2 - s1

        comparison += f"\n**Common Skills:** {', '.join(common) if common else 'None'}\n"
        comparison += f"**Only {n1}:** {', '.join(only1) if only1 else 'None'}\n"
        comparison += f"**Only {n2}:** {', '.join(only2) if only2 else 'None'}\n"

        # Verdict
        c1_score = c1.get('composite_score', 0)
        c2_score = c2.get('composite_score', 0)
        if c1_score > c2_score:
            comparison += f"\n**🏆 Verdict:** **{n1}** leads by {round(c1_score - c2_score, 1)} points."
        elif c2_score > c1_score:
            comparison += f"\n**🏆 Verdict:** **{n2}** leads by {round(c2_score - c1_score, 1)} points."
        else:
            comparison += f"\n**🏆 Verdict:** It's a tie! Consider domain fit and interview performance."

        return comparison

    def _handle_explain(self, message, results):
        """Explain why a candidate has a certain rank/score."""
        candidates = results.get('shortlisted', results.get('candidates', []))

        # Find the candidate being asked about
        numbers = re.findall(r'#?(\d+)', message)
        target = None

        if numbers:
            rank = int(numbers[0])
            for c in candidates:
                if c.get('final_rank') == rank:
                    target = c
                    break

        if not target:
            # Look for name mentions
            for c in candidates:
                name = c.get('name', '').lower()
                if name and name in message.lower():
                    target = c
                    break

        if not target and candidates:
            target = candidates[0]

        if not target:
            return "I couldn't identify which candidate you're asking about. Try **\"Why is #1 ranked first?\"**"

        name = target.get('name', target.get('filename', 'Unknown'))
        rank = target.get('final_rank', '?')

        explanation = f"## 💡 Why #{rank} {name} is ranked here\n\n"
        explanation += "### Score Breakdown\n"
        explanation += f"- **Composite Score:** {target.get('composite_score', 0)}/100\n"
        explanation += f"  - ATS (TF-IDF similarity): **{target.get('ats_score', 0)}** × 25% weight\n"
        explanation += f"  - Keyword Match: **{target.get('keyword_score', 0)}%** × 35% weight\n"
        explanation += f"  - LLM Analysis: **{target.get('llm_score', 0)}** × 40% weight\n\n"

        # Strengths & weaknesses
        explanation += "### Strengths\n"
        matched = target.get('keywords_matched', [])
        if matched:
            explanation += f"- ✅ Matched {len(matched)} keywords: {', '.join(matched[:6])}\n"
        if target.get('experience_years', 0) >= 3:
            explanation += f"- ✅ Solid experience: {target.get('experience_years')} years\n"
        skills = target.get('skills', [])
        if len(skills) >= 5:
            explanation += f"- ✅ Broad skill set: {len(skills)} skills detected\n"

        explanation += "\n### Weaknesses\n"
        missed = target.get('keywords_missed', [])
        if missed:
            explanation += f"- ❌ Missing {len(missed)} keywords: {', '.join(missed[:5])}\n"
        if target.get('experience_years', 0) < 2:
            explanation += f"- ❌ Limited experience: {target.get('experience_years', 0)} years\n"
        if target.get('keyword_score', 0) < 40:
            explanation += f"- ❌ Low keyword match: {target.get('keyword_score', 0)}%\n"

        # Reflection data if available
        reflection = target.get('reflection', {})
        if reflection and reflection.get('was_corrected'):
            explanation += f"\n### 🔄 Score Correction Applied\n"
            explanation += (
                f"The Reflection Agent corrected the LLM score from "
                f"**{reflection.get('original_score')}** → **{reflection.get('corrected_score')}** "
                f"due to {len(reflection.get('anomalies', []))} anomalies detected.\n"
            )

        # LLM analysis if available
        analysis = target.get('llm_analysis', '')
        if analysis and len(str(analysis)) > 20:
            explanation += f"\n### 🤖 Full AI Analysis\n{str(analysis)[:600]}"

        return explanation

    def _handle_rerank(self, message, results):
        """Re-rank candidates based on user preferences."""
        candidates = results.get('candidates', [])
        if not candidates:
            return "❌ No candidates available to re-rank."

        msg_lower = message.lower()

        # Parse preferences from message
        new_prefs = {}

        # Experience preference
        if any(w in msg_lower for w in ["experience", "years", "senior"]):
            if any(w in msg_lower for w in ["more", "higher", "important", "prioritize", "prefer", "value"]):
                new_prefs['experience_weight'] = 'high'
            elif any(w in msg_lower for w in ["less", "lower", "deemphasize"]):
                new_prefs['experience_weight'] = 'low'

        # Skills preference
        if any(w in msg_lower for w in ["skill", "technical", "coding"]):
            if any(w in msg_lower for w in ["more", "higher", "important", "prioritize", "prefer"]):
                new_prefs['skills_weight'] = 'high'

        # Education preference
        if any(w in msg_lower for w in ["education", "degree", "master", "bachelor", "phd"]):
            if any(w in msg_lower for w in ["required", "must", "important", "prefer"]):
                new_prefs['education_weight'] = 'high'
            elif any(w in msg_lower for w in ["not required", "not important", "optional", "doesn't matter"]):
                new_prefs['education_weight'] = 'low'

        # Remote preference
        if "remote" in msg_lower:
            if any(w in msg_lower for w in ["ok", "fine", "acceptable", "yes", "prefer"]):
                new_prefs['remote_ok'] = True
            elif any(w in msg_lower for w in ["no", "not", "office", "on-site", "onsite"]):
                new_prefs['remote_ok'] = False

        # Budget/salary preference
        salary_match = re.search(r'(\d+)\s*[-–to]\s*(\d+)\s*k?', msg_lower)
        if salary_match:
            new_prefs['budget_min'] = int(salary_match.group(1))
            new_prefs['budget_max'] = int(salary_match.group(2))

        # Min experience requirement
        exp_match = re.search(r'(?:minimum|at least|min)\s*(\d+)\s*(?:years?|yrs?)', msg_lower)
        if exp_match:
            new_prefs['min_experience'] = int(exp_match.group(1))

        self.preferences.update(new_prefs)

        # Apply re-ranking
        reranked = self._apply_preference_rerank(candidates, self.preferences)

        # Build response
        shortlist_count = results.get('shortlist_count', 5)
        new_shortlisted = reranked[:shortlist_count]

        response = "## 🎯 Re-Ranked Results\n\n"

        if new_prefs:
            response += "**Preferences applied:**\n"
            for k, v in new_prefs.items():
                label = k.replace('_', ' ').title()
                response += f"- {label}: **{v}**\n"
            response += "\n"
        else:
            response += (
                "I'll re-rank based on your message. You can be more specific:\n"
                "- *\"Prioritize experience over skills\"*\n"
                "- *\"Minimum 3 years experience\"*\n"
                "- *\"Education not important\"*\n"
                "- *\"Remote work is OK\"*\n\n"
            )

        response += "**Updated Rankings:**\n\n"
        for i, c in enumerate(new_shortlisted):
            old_rank = c.get('final_rank', '?')
            new_rank = i + 1
            name = c.get('name', c.get('filename', 'Unknown'))
            change = ""
            if isinstance(old_rank, int) and old_rank != new_rank:
                if new_rank < old_rank:
                    change = f" ⬆️ (+{old_rank - new_rank})"
                else:
                    change = f" ⬇️ (-{new_rank - old_rank})"

            response += (
                f"**#{new_rank} {name}**{change} — "
                f"Score: {c.get('adjusted_composite', c.get('composite_score', 0))}\n"
            )

        # Update results in place
        for i, c in enumerate(reranked):
            c['final_rank'] = i + 1
        results['candidates'] = reranked
        results['shortlisted'] = reranked[:shortlist_count]

        return response

    def _apply_preference_rerank(self, candidates, preferences):
        """Apply user preferences to re-score and re-rank candidates."""
        for c in candidates:
            base_composite = c.get('composite_score', 0)
            adjustment = 0

            # Experience weight
            exp = c.get('experience_years', 0) or 0
            if preferences.get('experience_weight') == 'high':
                adjustment += min(15, exp * 2)
            elif preferences.get('experience_weight') == 'low':
                adjustment -= min(10, exp)

            # Education weight
            education = (c.get('education', '') or '').lower()
            if preferences.get('education_weight') == 'high':
                if 'master' in education or 'phd' in education or 'doctorate' in education:
                    adjustment += 10
                elif 'bachelor' in education:
                    adjustment += 3
            elif preferences.get('education_weight') == 'low':
                pass  # No adjustment

            # Skills weight
            skills = c.get('skills', [])
            if preferences.get('skills_weight') == 'high':
                adjustment += min(10, len(skills))

            # Min experience filter
            min_exp = preferences.get('min_experience', 0)
            if min_exp > 0 and exp < min_exp:
                adjustment -= 30  # Heavy penalty

            c['adjusted_composite'] = round(max(0, min(100, base_composite + adjustment)), 2)

        # Sort by adjusted composite
        reranked = sorted(candidates, key=lambda x: x.get('adjusted_composite', 0), reverse=True)
        return reranked

    def _handle_filter(self, message, results):
        """Filter candidates based on user criteria."""
        candidates = results.get('candidates', [])
        if not candidates:
            return "❌ No candidates available to filter."

        msg_lower = message.lower()
        original_count = len(candidates)
        filtered = list(candidates)

        # Filter by minimum experience
        exp_match = re.search(r'(?:minimum|at least|min|less than|under)\s*(\d+)\s*(?:years?|yrs?)', msg_lower)
        if exp_match:
            min_exp = int(exp_match.group(1))
            if "less than" in msg_lower or "under" in msg_lower:
                filtered = [c for c in filtered if (c.get('experience_years', 0) or 0) < min_exp]
                action = f"Kept candidates with < {min_exp} years experience"
            else:
                filtered = [c for c in filtered if (c.get('experience_years', 0) or 0) >= min_exp]
                action = f"Kept candidates with ≥ {min_exp} years experience"

        # Filter by minimum score
        score_match = re.search(r'(?:score|composite)\s*(?:above|over|>|>=|at least)\s*(\d+)', msg_lower)
        if score_match:
            min_score = int(score_match.group(1))
            filtered = [c for c in filtered if c.get('composite_score', 0) >= min_score]
            action = f"Kept candidates with composite score ≥ {min_score}"

        removed_count = original_count - len(filtered)

        if removed_count == 0:
            return "No candidates were filtered out. All match your criteria! ✅"

        # Re-rank
        for i, c in enumerate(filtered):
            c['final_rank'] = i + 1

        shortlist_count = results.get('shortlist_count', 5)
        results['candidates'] = filtered
        results['shortlisted'] = filtered[:shortlist_count]

        response = f"## 🚫 Filter Applied\n\n"
        response += f"- **Removed:** {removed_count} candidates\n"
        response += f"- **Remaining:** {len(filtered)} candidates\n\n"
        response += "**Updated Top 5:**\n\n"

        for c in filtered[:5]:
            response += (
                f"**#{c.get('final_rank')} {c.get('name', c.get('filename', 'Unknown'))}** — "
                f"Composite: {c.get('composite_score', 0)} | "
                f"Exp: {c.get('experience_years', '?')} yrs\n"
            )

        return response

    def _handle_summary(self, message, results):
        """Generate a concise summary of the results."""
        shortlisted = results.get('shortlisted', [])
        if not shortlisted:
            return "❌ No candidates available. Please run the pipeline first."

        response = "## 📊 Quick Summary\n\n"

        # Top picks
        response += "### 🏆 Top 3 Picks\n"
        for c in shortlisted[:3]:
            response += (
                f"**#{c.get('final_rank', '?')} {c.get('name', c.get('filename', 'Unknown'))}** — "
                f"Score: {c.get('composite_score', 0)} | "
                f"{c.get('domain', 'General')} | "
                f"{c.get('experience_years', '?')} yrs exp\n"
            )

        # Score distribution
        if shortlisted:
            scores = [c.get('composite_score', 0) for c in shortlisted]
            response += f"\n### 📈 Score Range\n"
            response += f"- Highest: **{max(scores)}** | Lowest: **{min(scores)}** | "
            response += f"Average: **{round(sum(scores) / len(scores), 1)}**\n"

        # Common skills across top candidates
        all_skills = []
        for c in shortlisted[:5]:
            all_skills.extend(s.lower() for s in c.get('skills', []))
        if all_skills:
            from collections import Counter
            top_skills = Counter(all_skills).most_common(5)
            response += f"\n### 🔧 Most Common Skills\n"
            for skill, count in top_skills:
                response += f"- **{skill.title()}** — found in {count}/{len(shortlisted[:5])} candidates\n"

        # Recommendation
        if shortlisted:
            best = shortlisted[0]
            response += f"\n### 🎯 My Recommendation\n"
            response += (
                f"Interview **{best.get('name', 'the top candidate')}** first. "
                f"They have the highest composite score ({best.get('composite_score', 0)}) "
                f"with {best.get('experience_years', '?')} years of experience "
                f"in {best.get('domain', 'General')}."
            )

        return response

    def _handle_candidate_query(self, message, results):
        """Handle queries about a specific candidate."""
        candidates = results.get('candidates', results.get('shortlisted', []))
        target = None

        # Try rank number
        numbers = re.findall(r'#?(\d+)', message)
        if numbers:
            rank = int(numbers[0])
            for c in candidates:
                if c.get('final_rank') == rank:
                    target = c
                    break

        # Try name match
        if not target:
            for c in candidates:
                name = c.get('name', '').lower()
                if name and name in message.lower():
                    target = c
                    break

        if not target:
            return "I couldn't find that candidate. Try **\"Tell me about #1\"** or use the candidate's name."

        return f"## 👤 Candidate Profile\n\n{self._format_candidate_summary(target)}"

    def _handle_general_with_llm(self, message, results):
        """Use LLM to answer general questions when Ollama is available."""
        if not self.llm_agent:
            return None

        shortlisted = results.get('shortlisted', [])
        candidates_summary = ""
        for c in shortlisted[:5]:
            candidates_summary += (
                f"#{c.get('final_rank')} {c.get('name', 'Unknown')}: "
                f"Composite={c.get('composite_score', 0)}, "
                f"Skills={c.get('skills', [])[:5]}, "
                f"Exp={c.get('experience_years', '?')}yrs, "
                f"Domain={c.get('domain', 'General')}\n"
            )

        # Include conversation context
        history_text = ""
        for entry in self.conversation_history[-4:]:
            role = entry.get('role', 'unknown')
            content = entry.get('content', '')[:200]
            history_text += f"{role}: {content}\n"

        prompt = f"""You are CognifyX AI Assistant, a helpful recruitment analyst.
You have just screened resumes and shortlisted candidates.

Top Candidates:
{candidates_summary}

Recent Conversation:
{history_text}

User asks: {message}

Provide a helpful, concise answer about the candidates or recruitment process.
Keep it under 200 words. Use markdown formatting."""

        response, elapsed = self.llm_agent.call_llm(
            self.llm_agent.MODELS.get("insights", "llama3:latest"),
            prompt,
            timeout=30
        )
        return response if response else None

    def chat(self, message, results=None):
        """
        Main entry point for handling a chat message.

        Args:
            message: User's chat message
            results: Current pipeline results dict (from session_state)

        Returns:
            str: AI response in markdown format
        """
        # Store in history
        self.conversation_history.append({
            "role": "user",
            "content": message
        })

        # Classify intent
        intent = self._classify_intent(message)

        # Route to handler
        if intent == "greeting":
            response = self._handle_greeting(message, results)
        elif not results or not results.get('shortlisted'):
            response = (
                "⏳ No pipeline results available yet. Please:\n"
                "1. Upload resume files in the sidebar\n"
                "2. Enter a Job Description and Keywords\n"
                "3. Click **🚀 Start AI Screening**\n\n"
                "Then come back and ask me anything about the results!"
            )
        elif intent == "compare":
            response = self._handle_compare(message, results)
        elif intent == "explain":
            response = self._handle_explain(message, results)
        elif intent == "rerank":
            response = self._handle_rerank(message, results)
        elif intent == "filter":
            response = self._handle_filter(message, results)
        elif intent == "summary":
            response = self._handle_summary(message, results)
        elif intent == "candidate_query":
            response = self._handle_candidate_query(message, results)
        else:
            # Try LLM for general queries
            llm_response = self._handle_general_with_llm(message, results)
            if llm_response:
                response = llm_response
            else:
                response = self._handle_general_fallback(message, results)

        # Store response in history
        self.conversation_history.append({
            "role": "assistant",
            "content": response
        })

        return response

    def _handle_general_fallback(self, message, results):
        """Fallback for unrecognized queries."""
        return (
            "I'm not sure what you're asking. Here are things I can help with:\n\n"
            "- 🔍 **\"Compare #1 and #3\"** — Side-by-side comparison\n"
            "- 💬 **\"Why is #2 ranked higher?\"** — Explain scoring reasoning\n"
            "- 🎯 **\"Prioritize experience\"** — Re-rank with your preferences\n"
            "- 🚫 **\"Remove candidates with < 2 years\"** — Filter results\n"
            "- 📝 **\"Summary\"** — Quick overview of results\n"
            "- 👤 **\"Tell me about #1\"** — Deep dive into a candidate\n"
            "- 📊 **\"Compare top 2\"** — Auto-compare top ranked candidates"
        )

    def get_proactive_questions(self, results):
        """
        Generate smart proactive questions based on candidate analysis.
        Called after pipeline completion to initiate conversation.
        """
        if not results or not results.get('shortlisted'):
            return None

        shortlisted = results['shortlisted']
        questions = []

        # Check for experience gaps
        exp_values = [c.get('experience_years', 0) or 0 for c in shortlisted]
        if max(exp_values) - min(exp_values) > 5:
            questions.append(
                f"📊 Experience range is wide ({min(exp_values)}-{max(exp_values)} years). "
                f"Should I prioritize senior or junior candidates?"
            )

        # Check for low keyword matches in top candidates
        low_kw = [c for c in shortlisted[:3] if c.get('keyword_score', 0) < 40]
        if low_kw:
            names = ", ".join(c.get('name', c.get('filename', '?'))[:20] for c in low_kw)
            questions.append(
                f"⚠️ {len(low_kw)} of your top 3 have low keyword matches ({names}). "
                f"Should I weight keyword match more heavily?"
            )

        # Check for domain diversity
        domains = set(c.get('domain', 'General') for c in shortlisted)
        if len(domains) > 2:
            questions.append(
                f"🏷️ Candidates span {len(domains)} domains ({', '.join(list(domains)[:3])}). "
                f"Want me to focus on a specific domain?"
            )

        # Check if scores are very close
        if shortlisted:
            scores = [c.get('composite_score', 0) for c in shortlisted[:3]]
            if len(scores) >= 2 and max(scores) - min(scores) < 5:
                questions.append(
                    "🤝 Your top candidates have very similar scores. "
                    "Want me to break the tie with a detailed comparison?"
                )

        # Education diversity
        edu_types = set(
            c.get('education', 'Not specified') for c in shortlisted
            if c.get('education') and c.get('education') != 'Not specified'
        )
        if len(edu_types) > 1:
            questions.append(
                f"🎓 Education levels vary ({', '.join(edu_types)}). "
                f"Is a specific degree required or preferred?"
            )

        return questions[:3]  # Return max 3 questions
