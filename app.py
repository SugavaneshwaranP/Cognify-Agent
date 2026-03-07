"""
CognifyX – AI Resume Intelligence System
Entry point for CLI mode.
"""
from agents.coordinator_agent import CoordinatorAgent
import json


def main():
    print("=" * 60)
    print("  CognifyX – AI Resume Intelligence System")
    print("=" * 60)

    coordinator = CoordinatorAgent(dataset_path="dataset/resumes")

    jd = """
    We are looking for a Senior Software Engineer with:
    - 5+ years of experience in Python
    - Strong knowledge of SQL and Database design
    - Experience with LLMs and AI agent architectures
    - Knowledge of React and modern frontend frameworks
    - Experience with Docker, Kubernetes, and CI/CD
    """

    keywords = "python, sql, react, docker, kubernetes, llm, ai, machine learning, 5 years"

    results = coordinator.run_pipeline(jd, keywords_input=keywords, shortlist_count=5)

    if results:
        print("\n" + "=" * 60)
        print("  🏆 SHORTLISTED CANDIDATES")
        print("=" * 60)
        for c in results['shortlisted']:
            print(f"\n  #{c['final_rank']} – {c.get('name', c['filename'])}")
            print(f"     Composite: {c['composite_score']}  |  ATS: {c['ats_score']}  |  KW: {c['keyword_score']}  |  LLM: {c['llm_score']}")
            print(f"     Domain: {c.get('domain', 'N/A')}  |  Experience: {c.get('experience_years', '?')} yrs")

        print("\n" + "=" * 60)
        print("  🤖 FINAL AI INSIGHTS")
        print("=" * 60)
        print(results['final_summary'])

        print("\n" + "=" * 60)
        print("  ⏱️ TIMINGS")
        print("=" * 60)
        for stage, t in results['timings'].items():
            print(f"  {stage}: {t}s")
    else:
        print("Pipeline failed or no resumes found.")


if __name__ == "__main__":
    main()
