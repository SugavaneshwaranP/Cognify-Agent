"""
CognifyX – Coordinator Agent
Orchestrates the full resume screening pipeline with:
1. Parsing → 2. ATS Keyword Scoring → 3. LLM Extraction → 4. LLM Scoring → 5. Ranking → 6. Insights
"""
import json
import time

from pipeline.resume_parser import ResumeParser
from pipeline.ats_filter import ATSFilter
from pipeline.llm_extractor import LLMExtractor
from pipeline.candidate_ranker import CandidateRanker
from pipeline.final_analysis import FinalAnalysis
from agents.llm_agent import LLMAgent
from agents.reflection_agent import ReflectionAgent
from database.resume_db import ResumeDB


class CoordinatorAgent:
    def __init__(self, dataset_path="dataset/resumes", top_n=40):
        self.dataset_path = dataset_path
        self.parser = ResumeParser()
        self.ats_filter = ATSFilter(top_n=top_n)
        self.llm_agent = LLMAgent()
        self.extractor = LLMExtractor(self.llm_agent)
        self.ranker = CandidateRanker(self.llm_agent)
        self.analyzer = FinalAnalysis(self.llm_agent)
        self.reflection_agent = ReflectionAgent(self.llm_agent)
        self.db = ResumeDB()
        self.logs = []
        self.timings = {}

    def log(self, message):
        self.logs.append(message)
        print(f"[Coordinator] {message}")

    def run_pipeline(self, job_description, keywords_input="", shortlist_count=5):
        """
        Run the full pipeline.

        Args:
            job_description: The JD text
            keywords_input: Comma/newline separated keywords for ATS scoring
            shortlist_count: Number of candidates to shortlist (default 5)
        
        Returns:
            dict with top_candidates, final_summary, timings, logs
        """
        total_start = time.time()
        self.log("🚀 Starting CognifyX Pipeline...")

        # Check Ollama status
        ollama_up, available_models = self.llm_agent.check_ollama_status()
        if ollama_up:
            self.log(f"✅ Ollama is running. Available models: {', '.join(available_models[:5])}")
        else:
            self.log("⚠️ Ollama not detected. Using heuristic analysis mode.")

        # Create pipeline run in DB
        run_id = self.db.create_run(job_description, keywords_input, 0)

        # ── Stage 1: Parse ────────────────────────────────────────────
        t0 = time.time()
        self.log("📄 Stage 1: Parsing resumes from dataset...")
        resumes = self.parser.parse_directory(self.dataset_path)
        self.timings['parsing'] = round(time.time() - t0, 2)
        self.log(f"   Found {len(resumes)} valid resumes. ({self.timings['parsing']}s)")

        if not resumes:
            self.log("❌ No resumes found to process.")
            return None

        # Store resumes in DB
        for r in resumes:
            self.db.store_resume(r['filename'], r['text'])

        # ── Stage 2: ATS Keyword Filtering ────────────────────────────
        t0 = time.time()
        self.log("🔍 Stage 2: Running ATS Keyword + TF-IDF Filtering...")
        top_resumes = self.ats_filter.calculate_scores(resumes, job_description, keywords_input)
        self.timings['ats_filter'] = round(time.time() - t0, 2)
        self.log(f"   Filtered to top {len(top_resumes)} candidates. ({self.timings['ats_filter']}s)")

        # ── Stage 3: LLM Extraction (Qwen) ───────────────────────────
        t0 = time.time()
        self.log("🧠 Stage 3: Extracting structured data (Qwen)...")
        structured_profiles = []
        for i, resume in enumerate(top_resumes):
            # Check cache first
            cached_data = self.db.get_extraction(resume['text'])
            if cached_data:
                profile = cached_data
                self.log(f"   [Cache] {resume['filename']}")
            else:
                profile = self.extractor.extract(resume['text'], identifier=resume['filename'])
                self.db.save_extraction(resume['text'], profile)
                self.log(f"   [Extracted] {resume['filename']}")

            profile['filename'] = resume['filename']
            if not profile.get('name') or profile.get('name') == "Sample Candidate":
                profile['name'] = resume['filename'].replace('.pdf', '').replace('.docx', '').replace('_', ' ')

            profile['ats_score'] = resume.get('score', 0)
            profile['keyword_score'] = resume.get('keyword_score', 0)
            profile['keywords_matched'] = resume.get('keywords_matched', [])
            profile['keywords_missed'] = resume.get('keywords_missed', [])
            profile['tfidf_score'] = resume.get('tfidf_score', 0)
            profile['full_text'] = resume['text']
            structured_profiles.append(profile)

        self.timings['extraction'] = round(time.time() - t0, 2)
        self.log(f"   Extraction complete. ({self.timings['extraction']}s)")

        # ── Stage 4: LLM Scoring & Ranking (Mistral) ─────────────────
        t0 = time.time()
        self.log("📊 Stage 4: Scoring candidates (Mistral)...")
        for profile in structured_profiles:
            profile_brief = {k: v for k, v in profile.items() if k != 'full_text'}
            p_json = json.dumps(profile_brief, default=str)

            cached_score = self.db.get_scoring(job_description, p_json)
            if cached_score:
                analysis = cached_score
                self.log(f"   [Cache] Scoring for {profile['filename']}")
            else:
                analysis = self.ranker.rank(job_description, profile, identifier=profile['filename'])
                self.db.save_scoring(job_description, p_json, analysis)
                self.log(f"   [Scored] {profile['filename']}")

            profile['llm_analysis'] = analysis
            llm_score = CandidateRanker.extract_llm_score(analysis)
            profile['llm_score'] = llm_score
            profile['composite_score'] = CandidateRanker.compute_composite_score(
                profile['ats_score'], profile['keyword_score'], llm_score
            )

        self.timings['scoring'] = round(time.time() - t0, 2)
        self.log(f"   Scoring complete. ({self.timings['scoring']}s)")

        # ── Stage 4.5: Self-Correction & Reflection ───────────────────
        t0 = time.time()
        self.log("🔄 Stage 4.5: Running Self-Correction & Reflection...")
        reflection_results = self.reflection_agent.reflect_batch(
            structured_profiles, job_description, use_llm=True
        )

        # Apply corrected scores
        corrected_count = 0
        for profile, reflection in zip(structured_profiles, reflection_results):
            if reflection['was_corrected']:
                old_score = profile['llm_score']
                profile['llm_score'] = reflection['corrected_score']
                profile['composite_score'] = CandidateRanker.compute_composite_score(
                    profile['ats_score'], profile['keyword_score'], profile['llm_score']
                )
                profile['reflection'] = reflection
                corrected_count += 1
                self.log(
                    f"   ⚠️ {profile.get('name', profile['filename'])}: "
                    f"{old_score} → {reflection['corrected_score']} "
                    f"({len(reflection['anomalies'])} anomalies)"
                )
            else:
                profile['reflection'] = reflection
                self.log(f"   ✅ {profile.get('name', profile['filename'])}: Score validated.")

        # Forward reflection logs
        for log_msg in self.reflection_agent.reflection_logs:
            self.log(log_msg)

        self.timings['reflection'] = round(time.time() - t0, 2)
        self.log(
            f"   Reflection complete: {corrected_count}/{len(structured_profiles)} "
            f"scores corrected. ({self.timings['reflection']}s)"
        )

        # ── Stage 5: Final Ranking ────────────────────────────────────
        ranked_profiles = CandidateRanker.rank_candidates(structured_profiles)
        shortlisted = ranked_profiles[:shortlist_count]

        # Mark shortlisted
        for p in shortlisted:
            p['shortlisted'] = True
        for p in ranked_profiles[shortlist_count:]:
            p['shortlisted'] = False

        # Store scores in DB
        for p in ranked_profiles:
            text_hash = self.db._get_hash(p['full_text'])
            resume_id = self.db.get_resume_id(text_hash)
            if resume_id:
                self.db.store_score(
                    run_id, resume_id,
                    ats_score=p.get('ats_score', 0),
                    keyword_score=p.get('keyword_score', 0),
                    llm_score=p.get('llm_score', 0),
                    composite_score=p.get('composite_score', 0),
                    llm_analysis=str(p.get('llm_analysis', '')),
                    final_rank=p.get('final_rank', 0),
                    shortlisted=1 if p.get('shortlisted') else 0
                )
        self.db.update_run(run_id, len(shortlisted))

        # ── Stage 6: Final AI Insights ────────────────────────────────
        t0 = time.time()
        self.log("🤖 Stage 5: Generating Final AI Insights (LLaMA)...")
        top_data = [{k: v for k, v in p.items() if k != 'full_text'} for p in shortlisted]
        summary = self.analyzer.analyze(json.dumps(top_data, default=str)[:4000])
        self.timings['insights'] = round(time.time() - t0, 2)
        self.log(f"   Insights generated. ({self.timings['insights']}s)")

        total_time = round(time.time() - total_start, 2)
        self.timings['total'] = total_time
        self.log(f"✅ Pipeline complete in {total_time}s")

        return {
            "top_candidates": ranked_profiles,
            "shortlisted": shortlisted,
            "final_summary": summary,
            "reflection_results": reflection_results,
            "timings": self.timings,
            "logs": self.logs,
            "run_id": run_id
        }
