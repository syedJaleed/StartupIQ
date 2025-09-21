import os 
import sys
import json
from google import genai 
from dotenv import load_dotenv
from playwright.sync_api import sync_playwright, TimeoutError
from vertexai.preview.generative_models import GenerativeModel, Part

load_dotenv()

linkedin_email = 'rajarishi449@gmail.com'
linkedin_password = 'hanTZ123$$'
output_file = 'output.png'
PROJECT_NAME = os.getenv('PROJECT_NAME')
LOCATION = os.getenv('LOCATION')
MODEL = os.getenv('MODEL')
linkedin_profile_analysis_prompt = ("""
                    You are an advanced information extraction system.
                    Your task is to analyze the provided LinkedIn profile screenshot and return ONLY a strictly valid JSON object with no additional text, comments, or formatting outside JSON.
                    Extract the following fields if visible:
                    {
                    "Name": string,
                    "Headline": string,
                    "Location": string,
                    "About": string,
                    "Education": [ { "Institution": string, "Degree": string, "Field": string, "Dates": string } ],
                    "Experience": [ { "Title": string, "Company": string, "Dates": string, "Location": string, "Description": string } ],
                    "Skills": [string],
                    "Certifications": [string],
                    "Achievements": [string],
                    "Languages": [string],
                    "Recommendations": [ { "From": string, "Text": string } ],
                    "Contact": { "Email": string, "Phone": string, "Website": string, "Social": [string] }
                    }
                    Rules:
                    - If a field is not visible, output it as an empty string "" or empty list [].
                    - Preserve the hierarchical structure exactly as shown.
                    - Do not add explanations, extra keys, or text outside the JSON.
                    - Ensure the JSON is fully valid and parseable in Python.
            """)
analyse_single_person_prompt = ("""
                    Strictly follow the below details.
                    You are a highly analytical evaluation system for founders and startups.
                    You will be given two inputs:
                    1. Founder/Employee Details — structured JSON data parsed from a profile.
                    2. Company Details — a structured Python dictionary describing the company.
                    Your task is to perform a comprehensive evaluation based on these inputs.
                    Specifically:
                    - Assess the founder/employee's credibility, domain expertise, and career relevance.
                    - Evaluate the company's maturity, traction, funding potential, differentiation, and market opportunity.
                    - Judge how well the founder/employee's background aligns with the company's mission and execution ability.
                    - Highlight expertise, weaknesses, and any red flags.
                    - Provide a **score out of 10** for the founder/employee based on skills, experience, and role.
                    - Justify the score clearly with reasoning.
                    - Provide how trustable the person is ( based on the frequent company switching )
                    Output must be a **strict JSON object only**, following this exact schema:  
                    {
                    "name": string,
                    "position": string,
                    "expertise": [string],
                    "weakness": [string],
                    "reason_for_expertise": string,
                    "reason_for_weakness": string,
                    "summary_data": string,
                    "score_point": number,
                    "technical_relevance_score": number,
                    "business_alignment_score": number,
                    "growth_potential_score": number,
                    "stability_fit_score": number,
                    "reason_for_score": string,
                    "reason_for_technical_relevance_score": string,
                    "reason_for_business_alignment_score": string,
                    "reason_for_growth_potential_score": string,
                    "reason_for_stability_fit_score": string,
                    }
                    Rules:
                    - No extra text, no markdown, no emojis.
                    - Output must be strictly valid JSON and directly parseable in Python.
            """)
analyse_team_prompt = ("""
                    I will provide you with:
                    1. Images of employees working in a company:
                    2. Details of the company, including:
                    - Industry/domain
                    - Company size
                    - Key products/services
                    - Business goals/strategic focus
                    - Technology stack or operational model (if relevant)
                    Your task is to **analyze the team as a whole** and provide a structured assessment using **team-level metrics**, including:
                    **Team Metrics Categories:**
                    1. **Technical Coverage**
                    - Breadth of skills across the team
                    - Depth of expertise in key technologies/domains
                    - Presence of rare or strategic skills
                    - Gaps in technical capabilities relative to company goals
                    2. **Business Alignment**
                    - Relevance of collective experience to company's products/services
                    - Experience in the company's industry or market
                    - Team's track record in roles that directly impact business outcomes
                    3. **Growth Potential**
                    - Team adaptability to new technologies or roles
                    - Evidence of learning and development (certifications, promotions)
                    - Potential for internal promotion or leadership growth
                    4. **Team Stability & Fit**
                    - Average tenure in current and past roles
                    - Employee retention risk indicators
                    - Cultural/organizational fit signals (e.g., leadership alignment, mentorship)
                    5. **Collaboration & Network Influence**
                    - Cross-functional experience
                    - Exposure to strategic partnerships, client interactions
                    - Mentorship or leadership roles within the team
                    6. **Overall Team Score**
                    - Weighted evaluation of the team across Technical Coverage, Business Alignment, Growth Potential, Stability & Fit, and Collaboration & Influence
                    - Provide reasoning for the score
                    **Output Format:**
                    Provide a structured JSON/dictionary output with:
                    - team_summary: Overall summary of the team
                    - metrics:
                    - technical_coverage_score: 0-10
                    - reason_for_technical_coverage_score
                    - business_alignment_score: 0-10
                    - reason_for_business_alignment_score
                    - growth_potential_score: 0-10
                    - reason_for_growth_potential_score
                    - stability_fit_score: 0-10
                    - reason_for_stability_fit_score
                    - collaboration_influence_score: 0-10
                    - reason_for_collaboration_influence_score
                    - overall_team_score: 0-10
                    - reason_for_overall_team_score
                    Analyze the team holistically, considering each employee's profile, and identify:
                    - Strengths of the team
                    - Weaknesses or gaps
                    - Opportunities for improvement
"""
)

def generate_from_prompt_and_image(prompt: str, image_part: any) -> str:
    """
    Sends a prompt and an image to the GenerativeModel and returns the response.
    
    :param prompt: The text prompt to send to the model.
    :param image_part: The image (binary data wrapped in Part).
    :return: The model's response as a string.
    """
    try:
        client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION)
        model = GenerativeModel(MODEL)
        if image_part:
            response = model.generate_content([prompt, image_part])
            return response.text
    except Exception as err:
        print("Exception in generate_from_prompt_and_image", err)

def generate_from_prompt_and_image_list(prompt: str, image_files: list) -> str:
    """
    Sends a prompt and multiple images to the GenerativeModel and returns the response.
    
    :param prompt: The text prompt to send to the model.
    :param image_files: List of image file paths to send (will be converted to Part objects).
    :return: The model's response as a string.
    """
    try:
        client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION)
        model = GenerativeModel(MODEL)
        
        image_parts = []
        for file_path in image_files:
            with open(file_path, "rb") as f:
                image_parts.append(Part.from_data(f.read(), mime_type="image/png"))
        
        input_parts = [prompt] + image_parts if image_parts else [prompt]
        
        response = model.generate_content(input_parts)
        return response.text
    except Exception as err:
        print("Exception in generate_from_prompt_and_images:", err)
        return ""

def clean_data(output: str):
    cleaned = output.strip("`").replace("json\n", "", 1).strip("`")
    data = json.loads(cleaned)
    return data

def generate_from_prompt(prompt: str):
    """
    Sends a prompt to the GenarativeModel and returns the response
    """
    try:
        client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION)
        response = client.models.generate_content(model="gemini-2.0-flash-lite-001",contents=prompt)
        return response.text
    except Exception as err:
        print('Exception in generate_from_prompt', err)

def do_screenshot(linkedInUrl, file_name = "", multiple = False):
    try:
        if not file_name:
            file_name = output_file
        with sync_playwright() as p:    
            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            print("[*] Navigating to LinkedIn login page...")
            page.goto("https://www.linkedin.com/login")
            print("[*] Filling login form...")
            page.fill('input[name="session_key"]', linkedin_email)
            page.fill('input[name="session_password"]', linkedin_password)
            page.click('button[type="submit"]')
            try:
                print("[*] Waiting for LinkedIn feed page after login...")
                print('Current_url', page.url)
                if "/feed" not in page.url:
                    page.wait_for_url(" **/feed", timeout=60000)
                else:
                    print("[*] Already on the feed page")
                print("[+] Login successful!")
            except TimeoutError:
                print("[!] Login failed or took too long.")
                print("Current URL:", page.url)
                browser.close()
            print(f"[*] Navigating to target page: {linkedInUrl}")
            if multiple:
                out = []
                # for index, linkedIn_url in enumerate(target_profile_list):
                for index, url in enumerate(linkedInUrl):
                    file_name = 'profile_' + str(index)
                    page.goto(url)
                    try:
                        page.wait_for_load_state("networkidle", timeout=20000)
                    except TimeoutError:
                        print("[!] Timeout waiting for target page to load.")
                    page.screenshot(path=file_name, full_page=True)
                    out.append(file_name)
                return out
            else:
                page.goto(linkedInUrl)
                try:
                    page.wait_for_load_state("networkidle", timeout=20000)
                except TimeoutError:
                    print("[!] Timeout waiting for target page to load.")
                page.screenshot(path=file_name, full_page=True)   
                print('page saved')
                return file_name
    except Exception as err:
        print('Exceptin in function do_screenshot')

def analyze_single_person(target_profile, company_detail, target_details = {}, file_name = ""):
    """
    :params target_profile: LinkedIn profile of the founder/employee
    :params comapny_details: details of the company
    :params target_details: extra data about the target user (Optional)
    """
    if target_profile:
            if file_name:
                file = do_screenshot(target_profile, file_name=file_name)
            else:
                file = do_screenshot(target_profile)
            prompt = linkedin_profile_analysis_prompt
            with open(file, "rb") as f:
                image_part = Part.from_data(f.read(), mime_type="image/png")
            response = generate_from_prompt_and_image(prompt, image_part)
            with open('output.json', 'a') as f:
                print(response, file = f)
            prompt = analyse_single_person_prompt
            cleaned_founder_dict = clean_data(response)
            founder_json = json.dumps(cleaned_founder_dict, indent=2)
            company_json = json.dumps(company_detail, indent=2)
            full_prompt = f"""{prompt}
                            Founder/Employee Details:
                            {founder_json}
                            Company Details:
                            {company_json}
                        """
            output = generate_from_prompt(full_prompt)
            cleaned_output = clean_data(output)
            print(cleaned_output)
            with open('final_output.json', 'a') as f:
                f.write(json.dumps(cleaned_output) + "\n")
            return cleaned_output

def analyze_team(target_profile_list, company_detail):
    try:
        # image_files = []
        # for index, linkedIn_url in enumerate(target_profile_list):
        #     file_name = f"profile_{index}.png"
        #     file_path = do_screenshot(linkedIn_url, file_name=file_name)
        #     image_files.append(file_path)
        #     print(f"[INFO] Processed {linkedIn_url}, saved screenshot: {file_path}")
        files = do_screenshot(target_profile_list, multiple=True)
        client = genai.Client(vertexai=True, project=PROJECT_NAME, location=LOCATION)
        model = GenerativeModel(MODEL)

        input_parts = [analyse_team_prompt]
        for file_path in files:
            with open(file_path, "rb") as f:
                image_part = Part.from_data(f.read(), mime_type="image/png")
                input_parts.append(image_part)
        response = model.generate_content(input_parts)
        cleaned_output = clean_data(response)
        with open('team_analysis.json', 'a') as f:
            f.write(json.dumps(cleaned_output) + "\n")
        return response.text
    except Exception as err:
        print(f"[ERROR] Exception in analyze_team: {err}")
        return None


# if __name__ == "__main__":
#     company_detail = {
#         'name': 'ziniosa',
#         'about': "We're a team of fashion lovers and tech innovators, redefining the way India shops for authentic pre-loved luxury. From iconic handbags and wallets to belts, shoes, and more, our curated collection brings you genuine designer pieces that combine timeless style with sustainability.At the heart of it all is affordable luxury. Were your exclusive, pocket-friendly destination for high-end fashion, offering authentic, pre-loved treasures that elevate your wardrobe and support a more eco-conscious future."
#     }
#     target_profile = "https://www.linkedin.com/in/ashri-jaiswal/"
#     output = analyze_single_person(target_profile, company_detail)

if __name__ == "__main__":
    company_detail = {
        'name': 'ziniosa',
        'about': "At EverestIMS, we believe that true innovation lies in creating solutions that drive real change. We understand that you need more than just quick solutions - you need a custom blend of problem-solving brilliance, practical design, and reliable execution. We focus on creating products and services that meet immediate needs while pushingn businesses to outperform the competition."
    }
    target_profile_list = ["https://www.linkedin.com/in/rajarishi-b-5a1523224/", "https://www.linkedin.com/in/annamalai-pichaimuthu-226458197/"]
    output = analyze_team(target_profile_list, company_detail)
    print('******************************88')
    print(output)