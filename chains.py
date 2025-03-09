import os
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from dotenv import load_dotenv

load_dotenv()

class Chain:
    def __init__(self):
        self.llm = ChatGroq(temperature=0, groq_api_key=os.getenv("GROQ_API_KEY"), model_name="llama-3.3-70b-versatile")

    def extract_jobs(self, cleaned_text):
        prompt_extract = PromptTemplate.from_template(
            """
            ### SCRAPED TEXT FROM WEBSITE:
            {page_data}
            ### INSTRUCTION:
            The scraped text is from the career's page of a website.
            Your job is to extract the job postings and return them in JSON format containing the following keys: `role`, `experience`, `skills`, and `description`.
            Only return the valid JSON.
            ### VALID JSON (NO PREAMBLE):
            """
        )
        chain_extract = prompt_extract | self.llm
        res = chain_extract.invoke(input={"page_data": cleaned_text})
        try:
            json_parser = JsonOutputParser()
            res = json_parser.parse(res.content)
        except OutputParserException:
            raise OutputParserException("Context too big. Unable to parse jobs.")
        return res if isinstance(res, list) else [res]

    def write_mail(self, job, links, username="Thamani", client_name="Hiring Manager", email_style="Formal"):
        """Generates cold emails based on the selected style."""
        
        style_instructions = {
            "Formal": "Maintain a professional and polished tone. Focus on achievements and qualifications.",
            "Casual": "Use a friendly, engaging tone. Keep it light while still showcasing strengths.",
            "Persuasive": "Be compelling and assertive. Highlight why you are the perfect fit with strong language."
        }

        # Convert style name to lowercase for prompt clarity
        email_style_lower = email_style.lower()

        prompt_email = PromptTemplate.from_template(
            """
            ### JOB DETAILS:
            - Role: {job_title}
            - Company: {company_name}
            - Experience Required: {experience}
            - Skills: {skills}
            - Description: {job_description}
            
            ### INSTRUCTION:
            You are {username}, a motivated MCA graduate with strong technical and analytical skills, seeking an opportunity to contribute to {company_name} as a {job_title}.
            Craft a **{email_style} cold email** to {client_name} demonstrating your skills, projects, and value.
            
            Style Instruction: {style_instruction}
            
            Highlight relevant projects, certifications, or portfolio links: {link_list}.
            
            ### EMAIL (NO PREAMBLE):
            """  
        )
    
        chain_email = prompt_email | self.llm
    
        # Extract job details, handling missing fields
        job_title = job.get("role", "the position")
        company_name = job.get("company", "the company")
        experience = job.get("experience", "not specified")
        skills = ", ".join(job.get("skills", [])) or "not mentioned"
        job_description = job.get("description", "No description provided.")
    
        # Filter out empty links
        valid_links = [link for link in links if link]
        formatted_links = "\n".join(f"- {link}" for link in valid_links) if valid_links else "No portfolio links provided."
    
        # Generate email
        res = chain_email.invoke({
            "job_title": job_title,
            "company_name": company_name,
            "experience": experience,
            "skills": skills,
            "job_description": job_description,
            "link_list": formatted_links,
            "username": username,
            "client_name": client_name,
            "email_style": email_style_lower,  # Fixed Issue
            "style_instruction": style_instructions[email_style]
        })
    
        return res.content

if __name__ == "__main__":
    print(os.getenv("GROQ_API_KEY"))
