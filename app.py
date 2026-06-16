import streamlit as st
import pandas as pd
import numpy as np
import joblib

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="AI Career Recommendation System",
    page_icon="🎯",
    layout="wide"
)

# =========================
# CUSTOM CSS
# =========================

st.markdown("""
<style>

.main {
    padding-top: 1rem;
}

div[data-testid="stMetric"] {
    border: 1px solid #444;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL
# =========================

model = joblib.load("career_xgb.pkl")
encoder = joblib.load("career_encoder.pkl")

st.title(
    "🚀 Adaptive Career Recommendation System"
)

st.caption(
    "Discover the best career paths based on your skills, interests and technical strengths."
)

st.markdown("""
Discover the best career paths based on your
skills, interests, and technical strengths.

The system provides:

✅ Top 3 Career Recommendations

✅ Skill Gap Analysis

✅ Personalized Learning Roadmap

✅ Estimated Learning Timeline
""")

st.markdown("---")




career_requirements = {

    "Data Scientist": {
        "python_score": 80,
        "statistics_score": 85,
        "probability_score": 80,
        "machine_learning_score": 80,
        "projects_completed": 5
    },

    "ML Engineer": {
        "python_score": 85,
        "machine_learning_score": 85,
        "deep_learning_score": 80,
        "linux_score": 70,
        "projects_completed": 5
    },

    "AI Engineer": {
        "python_score": 85,
        "deep_learning_score": 85,
        "statistics_score": 75,
        "machine_learning_score": 80,
        "projects_completed": 5
    },

    "Backend Developer": {
        "python_score": 75,
        "sql_score": 80,
        "database_score": 80,
        "linux_score": 70,
        "docker_score": 70
    },

    "Frontend Developer": {
        "html_score": 85,
        "css_score": 85,
        "javascript_score": 85,
        "projects_completed": 4
    },

    "Full Stack Developer": {
        "html_score": 80,
        "css_score": 80,
        "javascript_score": 80,
        "sql_score": 75,
        "database_score": 75
    },

    "Cloud Engineer": {
        "linux_score": 80,
        "docker_score": 80,
        "cloud_score": 85
    },

    "Cybersecurity Engineer": {
        "linux_score": 75,
        "consistency_score": 75,
        "cloud_score": 60
    },

    "Software Engineer": {
        "python_score": 75,
        "java_score": 70,
        "projects_completed": 4,
        "consistency_score": 70
    }
}

career_roadmaps = {

    "Data Scientist": [
        ("Master Statistics", "2 Weeks"),
        ("Learn Advanced Pandas", "1 Week"),
        ("Learn Scikit-Learn", "3 Weeks"),
        ("Build Data Science Projects", "4 Weeks"),
        ("Learn Model Deployment", "2 Weeks")
    ],

    "ML Engineer": [
        ("Master Machine Learning", "3 Weeks"),
        ("Learn Deep Learning", "4 Weeks"),
        ("Build ML Projects", "4 Weeks"),
        ("Deploy Models", "2 Weeks"),
        ("Learn MLOps", "3 Weeks")
    ],

    "AI Engineer": [
        ("Deep Learning", "4 Weeks"),
        ("Computer Vision", "3 Weeks"),
        ("NLP", "3 Weeks"),
        ("Transformers", "4 Weeks"),
        ("Deployment", "2 Weeks")
    ],

    "Backend Developer": [
        ("REST APIs", "2 Weeks"),
        ("Database Design", "2 Weeks"),
        ("Docker", "2 Weeks"),
        ("Authentication", "1 Week"),
        ("Cloud Deployment", "2 Weeks")
    ],

    "Frontend Developer": [
        ("Advanced JavaScript", "2 Weeks"),
        ("React", "3 Weeks"),
        ("State Management", "1 Week"),
        ("Testing", "1 Week"),
        ("Portfolio Projects", "3 Weeks")
    ],

    "Full Stack Developer": [
        ("React", "3 Weeks"),
        ("Backend Development", "3 Weeks"),
        ("Databases", "2 Weeks"),
        ("Authentication", "1 Week"),
        ("Deployment", "2 Weeks")
    ],

    "Cloud Engineer": [
        ("Linux", "2 Weeks"),
        ("Docker", "2 Weeks"),
        ("AWS", "4 Weeks"),
        ("Kubernetes", "4 Weeks"),
        ("CI/CD", "2 Weeks")
    ],

    "Cybersecurity Engineer": [
        ("Networking", "3 Weeks"),
        ("Linux Security", "2 Weeks"),
        ("Web Security", "3 Weeks"),
        ("Pen Testing", "4 Weeks"),
        ("Security Certifications", "6 Weeks")
    ],

    "Software Engineer": [
        ("Data Structures", "3 Weeks"),
        ("Algorithms", "3 Weeks"),
        ("System Design", "3 Weeks"),
        ("Backend Development", "3 Weeks"),
        ("Projects", "4 Weeks")
    ]
}

career_duration = {
    "Data Scientist": "3-4 Months",
    "ML Engineer": "4-5 Months",
    "AI Engineer": "5-6 Months",
    "Backend Developer": "3 Months",
    "Frontend Developer": "2-3 Months",
    "Full Stack Developer": "4-5 Months",
    "Cloud Engineer": "4 Months",
    "Cybersecurity Engineer": "5 Months",
    "Software Engineer": "3 Months"
}
def recommend_careers(student_features, model, encoder, top_k=3):

    probs = model.predict_proba(student_features)

    top_idx = np.argsort(probs[0])[::-1][:top_k]

    recommendations = []

    for idx in top_idx:

        recommendations.append({
            "career": encoder.classes_[idx]
        })

    return recommendations


def get_skill_gaps(student_row, career):

    reqs = career_requirements[career]

    strengths = []
    gaps = []

    for skill, target in reqs.items():

        current = float(student_row.iloc[0][skill])

        if current >= target:

            strengths.append(
                (skill, current)
            )

        else:

            gaps.append({
                "skill": skill,
                "current": current,
                "target": target,
                "gap": round(target-current, 1)
            })

    return strengths, gaps


def get_readiness(gaps):

    if len(gaps) == 0:
        return "Ready 🚀"

    elif len(gaps) <= 2:
        return "Almost Ready ⭐"

    return "Needs Improvement 📚"

# =========================
# SIDEBAR INPUTS
# =========================

st.sidebar.title("📝 Student Profile")

st.sidebar.markdown(
    "Adjust your skills and interests to receive personalized career recommendations."
)

# -------------------------
# Programming Skills
# -------------------------

st.sidebar.subheader("💻 Programming")

python_score = st.sidebar.slider("Python", 0, 100, 50)
cpp_score = st.sidebar.slider("C++", 0, 100, 50)
java_score = st.sidebar.slider("Java", 0, 100, 50)
javascript_score = st.sidebar.slider("JavaScript", 0, 100, 50)

html_score = st.sidebar.slider("HTML", 0, 100, 50)
css_score = st.sidebar.slider("CSS", 0, 100, 50)

sql_score = st.sidebar.slider("SQL", 0, 100, 50)
database_score = st.sidebar.slider("Database", 0, 100, 50)

# -------------------------
# Math & AI
# -------------------------

st.sidebar.subheader("📊 Math & AI")

statistics_score = st.sidebar.slider("Statistics", 0, 100, 50)
probability_score = st.sidebar.slider("Probability", 0, 100, 50)
linear_algebra_score = st.sidebar.slider("Linear Algebra", 0, 100, 50)
calculus_score = st.sidebar.slider("Calculus", 0, 100, 50)

machine_learning_score = st.sidebar.slider(
    "Machine Learning", 0, 100, 50
)

deep_learning_score = st.sidebar.slider(
    "Deep Learning", 0, 100, 50
)

# -------------------------
# Systems
# -------------------------

st.sidebar.subheader("⚙️ Systems")

linux_score = st.sidebar.slider("Linux", 0, 100, 50)
docker_score = st.sidebar.slider("Docker", 0, 100, 50)
cloud_score = st.sidebar.slider("Cloud", 0, 100, 50)

# -------------------------
# Interests
# -------------------------

st.sidebar.subheader("❤️ Interests")

interest_ai = st.sidebar.checkbox("AI")

interest_data_science = st.sidebar.checkbox(
    "Data Science"
)

interest_web_dev = st.sidebar.checkbox(
    "Web Development"
)

interest_backend = st.sidebar.checkbox(
    "Backend Development"
)

interest_frontend = st.sidebar.checkbox(
    "Frontend Development"
)

interest_mobile = st.sidebar.checkbox(
    "Mobile Development"
)

interest_cloud = st.sidebar.checkbox(
    "Cloud Computing"
)

interest_cybersecurity = st.sidebar.checkbox(
    "Cybersecurity"
)

# -------------------------
# Activity
# -------------------------

st.sidebar.subheader("📈 Activity")

projects_completed = st.sidebar.slider(
    "Projects Completed",
    0,
    20,
    3
)

quiz_avg_score = st.sidebar.slider(
    "Quiz Average Score",
    0,
    100,
    70
)

consistency_score = st.sidebar.slider(
    "Consistency Score",
    0,
    100,
    70
)

predict_button = st.sidebar.button(
    "🚀 Generate Recommendations"
)
# =========================
# PREDICTION
# =========================

if predict_button:

    ml_signal = (
        python_score *
        machine_learning_score
    )

    backend_signal = (
        sql_score *
        database_score
    )

    ai_signal = (
        deep_learning_score *
        statistics_score
    )

    frontend_signal = (
        html_score *
        css_score
    )

    student = pd.DataFrame([{

        "python_score": python_score,
        "cpp_score": cpp_score,
        "java_score": java_score,
        "javascript_score": javascript_score,

        "html_score": html_score,
        "css_score": css_score,

        "sql_score": sql_score,
        "database_score": database_score,

        "statistics_score": statistics_score,
        "probability_score": probability_score,
        "linear_algebra_score": linear_algebra_score,
        "calculus_score": calculus_score,

        "machine_learning_score":
            machine_learning_score,

        "deep_learning_score":
            deep_learning_score,

        "linux_score": linux_score,
        "docker_score": docker_score,
        "cloud_score": cloud_score,

        "interest_ai":
            int(interest_ai),

        "interest_data_science":
            int(interest_data_science),

        "interest_web_dev":
            int(interest_web_dev),

        "interest_backend":
            int(interest_backend),

        "interest_frontend":
            int(interest_frontend),

        "interest_mobile":
            int(interest_mobile),

        "interest_cloud":
            int(interest_cloud),

        "interest_cybersecurity":
            int(interest_cybersecurity),

        "projects_completed":
            projects_completed,

        "quiz_avg_score":
            quiz_avg_score,

        "consistency_score":
            consistency_score,

        "ml_signal":
            ml_signal,

        "backend_signal":
            backend_signal,

        "ai_signal":
            ai_signal,

        "frontend_signal":
            frontend_signal
    }])

    recommendations = recommend_careers(
        student,
        model,
        encoder
    )

    st.header(
        "🏆 Top 3 Career Recommendations"
    )

    st.markdown(
        """
        Explore the recommended career paths below.
        The first recommendation is typically the
        strongest match.
        """
    )
    for rank, rec in enumerate(
        recommendations,
        start=1
    ):

        career = rec["career"]

        strengths, gaps = get_skill_gaps(
            student,
            career
        )

        readiness = get_readiness(gaps)

        with st.expander(
            f"🏆 #{rank} {career}",
            expanded=(rank == 1)
        ):

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Career Readiness",
                    readiness
                )

            with col2:

                st.metric(
                    "Estimated Learning Time",
                    career_duration[career]
                )

            # =====================
            # SMART FEEDBACK
            # =====================

            if len(gaps) == 0:

                st.success(
                    f"""
                    Excellent!

                    You already satisfy the core skill
                    requirements for becoming a
                    {career}.

                    Focus on building projects,
                    internships, open-source
                    contributions and keeping your
                    skills up to date.
                    """
                )

            elif len(gaps) <= 2:

                st.info(
                    f"""
                    You are very close to becoming a
                    {career}.

                    Improve the highlighted skills
                    below and complete a few
                    portfolio projects.
                    """
                )

            else:

                st.warning(
                    f"""
                    You have strong potential for a
                    {career}, but multiple important
                    skills still need improvement.

                    Follow the roadmap below to
                    strengthen your profile.
                    """
                )

            # =====================
            # STRENGTHS
            # =====================

            st.subheader("✅ Your Strengths")

            if strengths:

                for skill, score in strengths:

                    st.success(
                        f"{skill.replace('_', ' ').title()} "
                        f"({score:.0f}/100)"
                    )

            else:

                st.info(
                    "No major strengths identified for this role yet."
                )

            # =====================
            # SKILL GAPS
            # =====================

            st.subheader("📈 Skill Gaps")

            if len(gaps) == 0:

                st.success(
                    """
                    No major skill gaps detected.

                    Continue improving your existing
                    strengths and gain practical
                    experience through projects.
                    """
                )

            else:

                for gap in gaps:

                    completion = min(
                        int(
                            (gap["current"] /
                             gap["target"]) * 100
                        ),
                        100
                    )

                    st.write(
                        f"**{gap['skill'].replace('_', ' ').title()}**"
                    )

                    st.progress(completion)

                    st.caption(
                        f"Current: {gap['current']:.0f} | "
                        f"Target: {gap['target']} | "
                        f"Need +{gap['gap']}"
                    )

            # =====================
            # ROADMAP
            # =====================

            st.subheader("🛣️ Learning Roadmap")

            roadmap = career_roadmaps[career]

            for i, (step, duration) in enumerate(
                roadmap,
                start=1
            ):

                st.write(
                    f"**Step {i}** — {step}"
                )

                st.caption(
                    f"Estimated Time: {duration}"
                )

            # =====================
            # FINAL ADVICE
            # =====================

            st.subheader("🎯 Recommendation")

            if len(gaps) == 0:

                st.success(
                    f"""
                    You are currently well-aligned
                    with the {career} career path.

                    Focus on advanced projects,
                    internships and interview
                    preparation.
                    """
                )

            elif len(gaps) <= 2:

                st.info(
                    f"""
                    You are very close to becoming
                    job-ready for a {career} role.

                    Address the remaining skill gaps
                    and build 2–3 strong projects.
                    """
                )

            else:

                st.warning(
                    f"""
                    Start by focusing on the largest
                    skill gaps shown above.

                    Consistent learning over the next
                    {career_duration[career]} can
                    significantly improve your
                    readiness for a {career} career.
                    """
                )

                