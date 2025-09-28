import streamlit as st
import json
import google.generativeai as genai
from deep_translator import GoogleTranslator
import random

# Tips
energy_tips = [
    "Switch to LED bulbs to save up to 80% electricity.",
    "Unplug chargers when not in use to prevent power wastage.",
    "Consider installing rooftop solar panels to cut electricity bills.",
    "Use energy-efficient appliances with a 5-star rating for better savings.",
    "Turn off fans and lights when leaving the room."
]


st.set_page_config(
    page_title="AI Civic Engagement Agent",
    page_icon="🇮🇳",
    layout="wide"
)

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") 

if GOOGLE_API_KEY and GOOGLE_API_KEY != "ApiKey":
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        llm = genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error configuring the AI model: {e}")
        st.stop()
else:
    st.error("🚨 Google API Key not found! Please paste your key into the `app.py` file on line 20.")
    st.stop()


@st.cache_data
def load_schemes_data():
    try:
        with open('data/schemes.json', 'r', encoding='utf-8') as f:
            data = json.load(f)
            return [scheme for scheme in data if scheme.get("scheme_name") != "Information not found"]
    except FileNotFoundError:
        st.error("`schemes.json` not found. Please run `scraper_agent.py` first to generate it, or create a default one.")
        return []

schemes_data = load_schemes_data()

def find_schemes_by_category(category: str) -> str:
    """Finds government schemes based on a category like 'Agriculture', 'Health', etc."""
    matching_schemes = [s['scheme_name'] for s in schemes_data if category.lower() in s.get('category', '').lower()]
    if not matching_schemes: return f"No schemes found for the category: {category}"
    return f"Found schemes for {category}: {', '.join(matching_schemes)}"

def get_scheme_details(scheme_name: str) -> str:
    """Gets all details for a single, specific government scheme by its name."""
    for scheme in schemes_data:
        if scheme_name.lower() in scheme.get('scheme_name', '').lower():
            return json.dumps(scheme)
    return f"Could not find details for a scheme named '{scheme_name}'."

MASTER_PROMPT = """
You are a helpful AI assistant for Indian citizens. Your goal is to answer user queries about government schemes by using the tools available to you.
Today's date is August 6, 2025.

**TOOLS:**
1. `find_schemes_by_category(category: str)`: Use to find schemes when asked about a general category (e.g., "health schemes", "schemes for farmers").
2. `get_scheme_details(scheme_name: str)`: Use when asked for details, eligibility, or application steps for a *specific* named scheme.
3. `FinalAnswer(answer: str)`: Use to provide a complete and final answer to the user, or if you don't need to use a tool.

**INSTRUCTIONS:**
1. Analyze the user's query.
2. Choose the best tool. If no tool is needed, use 'FinalAnswer'.
3. Respond ONLY in a valid JSON format: {"thought": "Your reasoning...", "tool": {"name": "tool_name", "args": {"arg_name": "value"}}}
"""

# --- AGENT ORCHESTRATOR ---
def run_agentic_loop(user_query, target_language='en'):
    """
    This function manages the agent's "reason-act" cycle.
    It takes a user's query and guides the AI to a final answer.
    """
    # Combine the master instructions with the user's specific query.
    prompt = f"{MASTER_PROMPT}\n\nUser Query: \"{user_query}\""
    
    try:
        # Call the Gemini LLM to get its plan in JSON format.
        response = llm.generate_content(prompt)
        action_json_str = response.text.strip().replace("```json", "").replace("```", "").strip()
        action = json.loads(action_json_str)
        
        #  Parse the AI's plan: its thought process and the tool it chose.
        thought = action.get("thought", "Thinking...")
        tool_name = action["tool"]["name"]
        tool_args = action["tool"]["args"]
        
        # Display the AI's reasoning to the user for transparency.
        st.session_state.thought = f"Agent reasoning: {thought}"

        # Execute the chosen tool with the provided arguments.
        if tool_name == "find_schemes_by_category":
            observation = find_schemes_by_category(**tool_args)
        elif tool_name == "get_scheme_details":
            observation = get_scheme_details(**tool_args)
        elif tool_name == "FinalAnswer":
            final_response = tool_args.get("answer", "I hope this information was helpful!")
            if target_language != 'en':
                final_response = GoogleTranslator(source='auto', target=target_language).translate(final_response)
            return final_response
        else:
            observation = f"Unknown tool: {tool_name}"

        final_prompt = f"Based on the user's query '{user_query}', a tool was used which returned the following observation: '{observation}'. Formulate a final, friendly, and helpful response to the user in markdown format."
        final_response = llm.generate_content(final_prompt).text
        
        # Translate the final response if needed
        if target_language != 'en':
            final_response = GoogleTranslator(source='auto', target=target_language).translate(final_response)

        return final_response

    except Exception as e:
        # Fall back to a direct answer if something goes wrong
        fallback_text = f"An agent error occurred: {e}. Falling back to a direct answer."
        st.warning(fallback_text)
        direct_response = llm.generate_content(f"Please answer this user query about Indian government schemes: {user_query}").text
        if target_language != 'en':
            direct_response = GoogleTranslator(source='auto', target=target_language).translate(direct_response)
        return direct_response

st.title("AI Civic Engagement Agent")
st.markdown("Your personal guide to discovering and understanding Indian Government Schemes.")

tab1, tab2, tab3, tab4 = st.tabs(["Home", "Scheme Finder", "AI Agent Chat", "All Schemes"])

with tab1:
    st.header("Why This Project Matters")
    st.markdown("""
    Millions of Indians miss out on government benefits because they don’t know they exist or can’t understand the application process. Our AI Civic Engagement Agent makes governance accessible, transparent, and personalized for every citizen.
    
    This project supports UN **SDG 16 (Peace, Justice & Strong Institutions)** and **SDG 11 (Sustainable Cities & Communities)**.
    """)
    st.divider()

    st.header("Features at a Glance")
    st.markdown("""
    - **Personalized Scheme Finder:** Fill out a simple form with your details to get a curated list of schemes that match your specific profile.
    - **AI Assistant Chat:** Ask complex questions in plain English or Hindi and get clear, understandable answers from our intelligent agent.
    - **Step-by-step Guidance:** The agent provides simple instructions on eligibility criteria, required documents, and how to apply for each scheme.
    - **Auto-Updated Data:** An offline AI agent works to keep our scheme information current by sourcing data from official government portals.
    """)
    
    st.divider()

    st.header("How It Helps You")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(label="Step 1", value="Fill Details")
        st.write("Provide your basic information in the Scheme Finder tab.")
    with col2:
        st.metric(label="Step 2", value="Get Recommendations")
        st.write("Receive a personalized list of schemes you are eligible for.")
    with col3:
        st.metric(label="Step 3", value="Ask & Apply")
        st.write("Use the AI Chat to ask questions and get guidance on how to apply.")

with tab2:
    st.header("Find Schemes Based on Your Profile")
    st.markdown("Fill in your details below, and we'll filter our database to find relevant schemes for you.")

    indian_states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", 
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", 
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", 
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim", 
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand", 
        "West Bengal", "Andaman and Nicobar Islands", "Chandigarh", 
        "Dadra and Nagar Haveli and Daman and Diu", "Lakshadweep", 
        "Delhi (National Capital Territory of Delhi)", "Puducherry", "Ladakh", "Jammu and Kashmir"
    ]
    
    dynamic_categories = sorted(list(set(s['category'] for s in schemes_data if 'category' in s)))
    hardcoded_categories = ["Health", "Agriculture", "Education", "Social Justice", "Women & Child Development", "Rural Development", "Entrepreneurship","Electrical","Housing","Technology"]
    all_categories = sorted(list(set(dynamic_categories + hardcoded_categories)))

    with st.form("user_profile_form"):
        # NEW: Age and Income now start from 0
        age = st.number_input("Your Age", min_value=0, max_value=120, value=0, step=1)
        gender = st.selectbox("Your Gender", ["Any", "Male", "Female", "Others"])
        # NEW: State filter added
        state = st.selectbox("Your State", ["All"] + sorted(indian_states))
        income = st.number_input("Annual Family Income (₹)", min_value=0, max_value=10000000, value=0, step=10000)
        category = st.selectbox("Scheme Category of Interest", ["All"] + all_categories)
        
        submitted = st.form_submit_button("Find My Schemes")



    if submitted:
        st.subheader("Recommended Schemes For You")

            # If user selected Electrical category, show a random energy tip
        if category == "Electrical":
            tip = random.choice(energy_tips)
            st.info(f"💡 Energy Tip: {tip}")

        
        filtered_schemes = []
        # Loop thrpugh every schemes in our database
        for scheme in schemes_data:
            try:
                # Safely convert data from the JSON file to numbers for comparison.
                min_age = int(scheme.get('min_age', 0))
                max_age = int(scheme.get('max_age', 100))
                min_income = int(scheme.get('min_income', 0))
                max_income = int(scheme.get('max_income', 5000000))

                # Check if the user's details match the scheme's requirements.
                age_match = min_age <= age <= max_age
                gender_match = scheme['target_gender'] == 'Any' or gender == 'Any' or scheme['target_gender'] == gender
                state_match = scheme['target_state'] == 'All' or state == 'All' or scheme['target_state'] == state
                income_match = min_income <= income <= max_income
                category_match = category == 'All' or scheme['category'] == category

                # If all criteria match, add the scheme to the results list.
                if age_match and gender_match and state_match and income_match and category_match:
                    filtered_schemes.append(scheme)
            except (ValueError, TypeError):
                # If any data is corrupted (e.g., text in a number field), skip this scheme.
                continue

        if not filtered_schemes:
            st.warning("No schemes found matching your specific criteria. Please try adjusting the filters.")
        else:
            for scheme in filtered_schemes:
                with st.container(border=True):
                    st.subheader(scheme['scheme_name'])
                    st.caption(f"Category: {scheme['category']} | For: {scheme['target_gender']} | State: {scheme['target_state']}")
                    st.markdown(f"**Description:** {scheme.get('description', 'N/A')}")
                    st.link_button("Visit Official Page", scheme.get('official_link', '#'))
                st.write("")

with tab3:
    st.header("Chat With Our AI Agent")
    lang_options = {'English': 'en', 'हिंदी (Hindi)': 'hi'}
    selected_lang_name = st.selectbox("Select Language:", options=lang_options.keys())
    target_lang_code = lang_options[selected_lang_name]

    st.markdown("Ask me anything about the available schemes.")

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "thought" not in st.session_state:
        st.session_state.thought = ""

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if st.session_state.thought:
        st.info(st.session_state.thought)
        st.session_state.thought = ""

    if prompt := st.chat_input("Your question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Agent is thinking..."):
                response = run_agentic_loop(prompt, target_lang_code)
                st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

with tab4:
    st.header("All Available Schemes")
    st.write("A complete list of all the government schemes currently in our knowledge base.")
    
    if not schemes_data:
        st.warning("No scheme data found. Please run the `scraper_agent.py` to populate this list.")
    else:
        for scheme in schemes_data:
            with st.container(border=True):
                st.subheader(scheme.get('scheme_name', 'Unnamed Scheme'))
                st.markdown(f"**Description:** {scheme.get('description', 'Information not available.')}")
                st.link_button("Apply Here / Official Page ↗️", scheme.get('official_link', '#'))
            st.write("") 

