# llm_chain_utils.py
from langchain import LLMChain, PromptTemplate
from langchain.llms import OpenAI  # or Gemini wrapper if available

def compare_resume_jd(jd_text: str, resume_text: str, llm_instance):
    template = (
        "You are an HR recruiter. Compare the following job description (JD) "
        "with the candidate's resume. Highlight matching skills, missing skills, "
        "and give a relevance score (0–100).\n\n"
        "JD:\n{jd}\n\nResume:\n{resume}"
    )
    prompt = PromptTemplate(template=template, input_variables=["jd", "resume"])
    chain = LLMChain(llm=llm_instance, prompt=prompt)
    return chain.run({"jd": jd_text, "resume": resume_text})
