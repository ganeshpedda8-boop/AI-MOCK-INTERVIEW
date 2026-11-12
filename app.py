
import streamlit as st
import ollama
from google import genai
import json
import re
import html

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="AI Mock Interview Assistant",
    page_icon="🎯",
    layout="wide"
)

# --------------------------------------------------
#

# --------------------------------------------------
# CUSTOM DESIGN
# --------------------------------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b1220;
    color: white;
}

h1, h2, h3, p, label {
    color: white !important;
}

div[data-testid="stAlert"] {
    background-color: #1e293b !important;
    border: 1px solid #3b82f6 !important;
    color: white !important;
}

div[data-testid="stAlert"] p,
div[data-testid="stAlert"] div,
div[data-testid="stAlert"] span {
    color: white !important;
}

.stButton > button {
    background-color: #2563eb;
    color: white !important;
    border-radius: 10px;
    border: none;
    padding: 10px 20px;
    font-weight: bold;
}

.stButton > button:hover {
    background-color: #1d4ed8;
}

div[data-testid="stTextArea"] textarea {
    background-color: #172033 !important;
    color: white !important;
}

div[data-baseweb="select"] > div {
    background-color: #172033;
    color: white;
}

section[data-testid="stSidebar"] {
    background-color: #111827;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------

OLLAMA_MODEL = "llama3.2"
GEMINI_MODEL = "gemini-3.8-flash"

try:
    api_key = st.secrets.get("GEMINI_API_KEY", "")
    gemini_client = genai.Client(api_key=api_key) if api_key else None
except Exception:
    gemini_client = None


# --------------------------------------------------
# JSON HELPER
# --------------------------------------------------

def extract_json(text):
    """Extract JSON safely from an AI response."""

    text = text.strip()

    # Remove Markdown code fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)

        if match:
            return json.loads(match.group(0))

        raise ValueError("The AI returned an invalid JSON response.")


# --------------------------------------------------
# OLLAMA FUNCTION
# --------------------------------------------------

def call_ollama(prompt):

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        format="json",
        options={
            "temperature": 0.3
        }
    )

    return extract_json(response.message.content)


# --------------------------------------------------
# GEMINI FUNCTION
# --------------------------------------------------

def call_gemini(prompt):

    if gemini_client is None:
        raise RuntimeError(
            "Gemini API key is missing. Add GEMINI_API_KEY "
            "to .streamlit/secrets.toml."
        )

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "temperature": 0.3
        }
    )

    if not response.text:
        raise ValueError("Gemini returned an empty response.")

    return extract_json(response.text)


# --------------------------------------------------
# COMMON AI FUNCTION
# --------------------------------------------------

def ask_ai(prompt, provider):

    if provider == "Ollama":
        return call_ollama(prompt), "Ollama"

    if provider == "Gemini":
        return call_gemini(prompt), "Gemini"

    # Auto mode: Gemini first, Ollama fallback
    try:
        result = call_gemini(prompt)
        return result, "Gemini"

    except Exception as gemini_error:

        st.warning(
            "Gemini is unavailable. Switching to Ollama. "
            f"Reason: {gemini_error}"
        )

        result = call_ollama(prompt)
        return result, "Ollama (Fallback)"


# --------------------------------------------------
# INTERVIEW QUESTION GENERATION
# --------------------------------------------------

def generate_questions(role, level, interview_type, provider):

    prompt = f"""
You are an expert technical interviewer.

Create exactly 5 interview questions.

Candidate details:
Job role: {role}
Experience level: {level}
Interview type: {interview_type}

Instructions:
- Questions must match the candidate's experience.
- Use clear and professional language.
- Avoid duplicate questions.
- Include practical questions when appropriate.
- Follow the selected interview type.

Return only valid JSON in this format:

{{
    "questions": [
        "Question 1",
        "Question 2",
        "Question 3",
        "Question 4",
        "Question 5"
    ]
}}
"""

    result, used_provider = ask_ai(prompt, provider)

    questions = result.get("questions", [])

    if not isinstance(questions, list) or len(questions) != 5:
        raise ValueError("AI did not generate exactly 5 questions.")

    return questions, used_provider


# --------------------------------------------------
# 
def evaluate_answer(question, answer, role, level, provider):

    prompt = f"""
You are a strict and fair professional interview evaluator.

Evaluate the candidate's answer carefully.

JOB ROLE: {role}
EXPERIENCE LEVEL: {level}

INTERVIEW QUESTION:
{question}

CANDIDATE ANSWER:
{answer}

IMPORTANT EVALUATION RULES:

1. First, understand exactly what the question is asking.
2. Check whether the candidate's answer addresses that question.
3. Never give marks just because the answer contains Python
   terminology or technically correct information unrelated
   to the question.
4. If the answer is completely unrelated, give 0 to 2 marks.
5. If the answer is partially relevant but has major errors,
   give 3 to 4 marks.
6. If the answer is partly correct but incomplete, give 5 to 6 marks.
7. If the answer is mostly correct with minor mistakes,
   give 7 to 8 marks.
8. Give 9 to 10 marks only for excellent, accurate and complete
   answers.
9. Evaluate code for correctness, not just its appearance.
10. Do not invent mistakes or missing points.
11. Judge the answer according to the candidate's experience level.

SCORING:
- Correctness: 40%
- Relevance: 30%
- Clarity: 15%
- Completeness: 15%

Return ONLY valid JSON in this format:

{{
    "score": 0,
    "relevance": "Relevant",
    "reason": "Explain why this score was awarded.",
    "strengths": [
        "Strength 1"
    ],
    "missing_points": [
        "Missing point 1"
    ],
    "improved_answer": "Provide a correct and complete sample answer."
}}

The score must be an integer between 0 and 10.

The relevance must be one of:
"Relevant", "Partially Relevant", "Irrelevant".

If the answer is irrelevant, clearly explain why and do not
award marks for unrelated technical knowledge.
"""

    result, used_provider = ask_ai(prompt, provider)

    score = int(result.get("score", 0))
    result["score"] = max(0, min(10, score))

    return result, used_provider

# --------------------------------------------------

def evaluate_answer(question, answer, role, level, provider):
    prompt = f"""
You are an expert interview evaluator.

Evaluate the candidate's interview answer.

Job role: {role}
Experience level: {level}

Interview question:
{question}

Candidate answer:
{answer}

Evaluate based on:
- Correctness
- Relevance
- Clarity
- Technical understanding
- Completeness

Give an integer score from 0 to 10.

Return only valid JSON in this format:

{{
    "score": 7,
    "relevance": "Relevant",
    "reason": "Explanation of the score",
    "strengths": [
        "Strength 1",
        "Strength 2"
    ],
    "missing_points": [
        "Missing point 1",
        "Missing point 2"
    ],
    "improved_answer": "A clear and better version of the answer."
}}

Do not give full marks for incomplete or incorrect answers.
"""

    result, used_provider = ask_ai(prompt, provider)

    score = int(result.get("score", 0))
    result["score"] = max(0, min(10, score))

    return result, used_provider


# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "questions" not in st.session_state:
    st.session_state.questions = []

if "evaluations" not in st.session_state:
    st.session_state.evaluations = []

if "interview_details" not in st.session_state:
    st.session_state.interview_details = {}

if "saved_answers" not in st.session_state:
    st.session_state.saved_answers = []

if "used_providers" not in st.session_state:
    st.session_state.used_providers = []


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

st.title("🎯 AI Mock Interview Assistant")

st.markdown(
    "Practice interviews, improve your answers, "
    "and build confidence with AI."
)

st.divider()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("⚙️ Interview Settings")

provider = st.sidebar.selectbox(
    "Choose AI Provider",
    [
        "Ollama",
        "Gemini",
        "Auto (Gemini first, Ollama fallback)"
    ]
)

role = st.sidebar.selectbox(
    "Select Job Role",
    [
        "Python Developer",
        "Web Developer",
        "Data Analyst",
        "AI/ML Engineer",
        "Cybersecurity Analyst"
    ]
)

level = st.sidebar.selectbox(
    "Experience Level",
    [
        "Beginner / Fresher",
        "Intermediate",
        "Experienced"
    ]
)

interview_type = st.sidebar.selectbox(
    "Interview Type",
    [
        "Mixed",
        "Technical",
        "HR"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Ollama runs locally on your computer. "
    "Gemini requires a valid API key and available quota."
)


# --------------------------------------------------
# GENERATE INTERVIEW
# --------------------------------------------------

st.header("1. Start Your Interview")

if st.button("🚀 Generate Interview Questions", use_container_width=True):

    with st.spinner("AI is preparing your interview questions..."):

        try:

            questions, used_provider = generate_questions(
                role,
                level,
                interview_type,
                provider
            )

            st.session_state.questions = questions
            st.session_state.evaluations = []
            st.session_state.saved_answers = []
            st.session_state.used_providers = []

            st.session_state.interview_details = {
                "role": role,
                "level": level,
                "interview_type": interview_type,
                "selected_provider": provider,
                "question_provider": used_provider
            }

            st.success(
                f"Five questions generated using {used_provider}!"
            )

        except Exception as error:
            st.error(f"Question generation failed: {error}")


# --------------------------------------------------
# 
# --------------------------------------------------
# DISPLAY QUESTIONS AND ANSWERS
# --------------------------------------------------


# --------------------------------------------------



    # --------------------------------------------------
    #     # 
# --------------------------------------------------
# DISPLAY QUESTIONS AND ANSWERS
# --------------------------------------------------

if st.session_state.questions:

    st.divider()
    st.header("2. Your Interview Questions")

    st.write(
        "Answer all five questions below. "
        "Try to explain your answers clearly."
    )

    answers = []

    for index, question in enumerate(st.session_state.questions):

        st.subheader(f"Question {index + 1}")

        st.markdown(
            f"""
            <div style="
                background-color: #172033 !important;
                color: #FFFFFF !important;
                padding: 20px;
                border-radius: 12px;
                border: 2px solid #3B82F6;
                font-size: 18px;
                font-weight: 600;
                line-height: 1.6;
                margin-bottom: 15px;
            ">
                {html.escape(question)}
            </div>
            """,
            unsafe_allow_html=True
        )

        answer = st.text_area(
            f"Your answer for Question {index + 1}",
            key=f"answer_{index}",
            height=120,
            placeholder="Type your answer here..."
        )

        answers.append(answer)
    #---------------------------------------------
    # EVALUATE ANSWERS
    # --------------------------------------------------

    if st.button("📊 Evaluate My Answers", use_container_width=True):

        if any(not answer.strip() for answer in answers):

            st.warning(
                "Please answer all five questions before evaluation."
            )

        else:

            evaluations = []
            used_providers = []

            with st.spinner(
                "AI is evaluating your answers. Please wait..."
            ):

                try:

                    details = st.session_state.interview_details

                    for index, question in enumerate(
                        st.session_state.questions
                    ):

                        result, used_provider = evaluate_answer(
                            question,
                            answers[index],
                            details["role"],
                            details["level"],
                            provider
                        )

                        evaluations.append(result)
                        used_providers.append(used_provider)

                    st.session_state.evaluations = evaluations
                    st.session_state.used_providers = used_providers
                    st.session_state.saved_answers = answers

                    st.success(
                        "Your interview evaluation is complete!"
                    )

                except Exception as error:

                    st.error(f"Evaluation failed: {error}")


# --------------------------------------------------
# RESULTS
# --------------------------------------------------

if st.session_state.evaluations:

    st.divider()
    st.header("3. Interview Results")

    evaluations = st.session_state.evaluations

    total_score = sum(
        item["score"] for item in evaluations
    )

    percentage = (total_score / 50) * 100

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Score", f"{total_score}/50")
    col2.metric("Percentage", f"{percentage:.1f}%")
    col3.metric("Questions", "5/5")

    st.progress(percentage / 100)

    if percentage >= 80:
        st.success("Excellent performance! Keep it up.")

    elif percentage >= 60:
        st.info("Good performance. Keep improving.")

    elif percentage >= 40:
        st.warning("You need more practice.")

    else:
        st.error(
            "Keep learning and practicing. You can improve!"
        )

    st.subheader("Individual Answer Evaluation")

    for index, result in enumerate(evaluations):

        with st.expander(
            f"Question {index + 1} — Score: {result['score']}/10",
            expanded=True
        ):

            st.write("**Question:**")
            st.write(st.session_state.questions[index])

            st.write("**Your Answer:**")
            st.write(st.session_state.saved_answers[index])

            st.write("**Relevance:**")
            st.write(result.get("relevance", "Not provided"))

            st.write("**Reason:**")
            st.write(result.get("reason", "Not provided"))

            st.write("**Strengths:**")

            for point in result.get("strengths", []):
                st.write(f"✅ {point}")

            st.write("**Missing Points:**")

            for point in result.get("missing_points", []):
                st.write(f"⚠️ {point}")

            st.write("**Improved Answer:**")

            st.success(
                result.get("improved_answer", "Not provided")
            )

            if index < len(st.session_state.used_providers):
                st.caption(
                    "Evaluated using: "
                    + st.session_state.used_providers[index]
                )

    # --------------------------------------------------
    # DOWNLOAD REPORT
    # --------------------------------------------------

    st.divider()
    st.header("4. Download Interview Report")

    report = {
        "interview_details": st.session_state.interview_details,
        "total_score": total_score,
        "maximum_score": 50,
        "percentage": round(percentage, 2),
        "results": []
    }

    for index, result in enumerate(evaluations):

        report["results"].append({
            "question": st.session_state.questions[index],
            "candidate_answer": st.session_state.saved_answers[index],
            "evaluation": result,
            "evaluation_provider": st.session_state.used_providers[index]
        })

    st.download_button(
        label="⬇️ Download Interview Report (JSON)",
        data=json.dumps(
            report,
            indent=4,
            ensure_ascii=False
        ),
        file_name="interview_report.json",
        mime="application/json",
        use_container_width=True
    )


# --------------------------------------------------

# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "AI Mock Interview Assistant | Built with Python, "
    "Streamlit, Ollama and Google Gemini"
)

