import streamlit as st
from langchain_community.document_loaders import WebBaseLoader
from chains import Chain
from portfolio import Portfolio
from utils import clean_text

# Initialize backend services
chain = Chain()
portfolio = Portfolio()

def create_streamlit_app(llm, portfolio, clean_text):
    st.set_page_config(layout="wide", page_title="Cold Email Generator", page_icon="📧")
    st.title("📧 Cold Mail Generator")

    # User Inputs
    url_input = st.text_input("Enter a Job URL:", value="https://atliq.keka.com/careers/jobdetails/51326")
    username = st.text_input("Your Name (Default: John)", value="John")
    client_name = st.text_input("Recipient's Name (Default: Hiring Manager)", value="Hiring Manager")

    # Dropdown to select number of emails to generate
    num_emails = st.selectbox("Select Number of Emails to Generate", options=[1, 2, 3], index=0)

    # Dropdown to select email style
    email_style = st.selectbox("Choose Email Style", options=["Formal", "Casual", "Persuasive"], index=0)

    submit_button = st.button("Generate Cold Emails")

    if submit_button:
        try:
            st.info("🔄 Extracting job details...")

            # Load and clean job description
            loader = WebBaseLoader([url_input])
            page_content = loader.load().pop().page_content
            cleaned_data = clean_text(page_content)

            # Extract job details
            jobs = llm.extract_jobs(cleaned_data)

            if not jobs:
                st.warning("⚠️ No job postings found. Please check the URL.")
                return

            # Process extracted jobs
            for idx, job in enumerate(jobs, start=1):
                job_title = job.get("role", "Unknown Role")
                experience = job.get("experience", "Not Specified")
                skills = ", ".join(job.get("skills", [])) if job.get("skills") else "Not Specified"
                job_description = job.get("description", "No description available.")

                # Display job details
                st.subheader(f"📌 Job {idx}: {job_title}")
                st.write(f"**Experience Required:** {experience}")
                st.write(f"**Skills:** {skills}")
                st.write(f"**Description:** {job_description}")
                st.divider()

                # Fetch relevant portfolio links
                skills_list = job.get("skills", [])
                portfolio_links = [link for link in portfolio.query_links(skills_list) if link]

                # Generate multiple cold emails based on user selection
                for i in range(num_emails):
                    st.subheader(f"📩 Email {i+1} ({email_style} Style)")
                    email = llm.write_mail(job, portfolio_links, username, client_name, email_style)

                    if email:
                        st.code(email, language="markdown")
                    else:
                        st.error("⚠️ Email generation failed.")

        except Exception as e:
            st.error(f"❌ An Error Occurred: {str(e)}")


if __name__ == "__main__":
    create_streamlit_app(chain, portfolio, clean_text)
