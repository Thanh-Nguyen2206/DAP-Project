"""
Enhanced Chatbot Module with OpenAI and Gemini Integration
Supports single chat and comparison mode
"""

import streamlit as st

# AI Libraries imports
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

def show_api_status():
    """Display API key status and provide clear options"""
    st.markdown("###  API Key Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        openai_confirmed = st.session_state.get('confirmed_openai_key', '')
        compare_openai_confirmed = st.session_state.get('confirmed_compare_openai_key', '')
        
        if openai_confirmed or compare_openai_confirmed:
            st.success(" OpenAI Key Confirmed")
        else:
            st.info("OpenAI Key Pending")
    
    with col2:
        gemini_confirmed = st.session_state.get('confirmed_gemini_key', '')
        compare_gemini_confirmed = st.session_state.get('confirmed_compare_gemini_key', '')
        
        if gemini_confirmed or compare_gemini_confirmed:
            st.success(" Gemini Key Confirmed")
        else:
            st.info("Gemini Key Pending")
    
    with col3:
        if st.button(" Clear All API Keys", type="secondary"):
            # Clear all API keys from session state
            keys_to_clear = [
                'confirmed_openai_key', 'confirmed_gemini_key',
                'confirmed_compare_openai_key', 'confirmed_compare_gemini_key'
            ]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.success(" All API keys cleared!")
            st.rerun()
    
    st.markdown("---")

def render_enhanced_chatbot(current_context):
    """Main enhanced chatbot interface"""
    st.header(" AI Financial Assistant")
    
    # Show API Key Status
    show_api_status()
    
    # AI Chatbot selection
    ai_mode = st.selectbox(
        " Select AI Model",
        ["Single Chat (OpenAI)", "Single Chat (Gemini)", "Compare Models (OpenAI vs Gemini)"],
        help="Choose between OpenAI GPT, Google Gemini, or compare both models side by side"
    )
    
    if ai_mode == "Single Chat (OpenAI)":
        st.subheader(" Chat with OpenAI GPT-4o")
        
        # Initialize chat history for OpenAI
        if 'openai_chat_history' not in st.session_state:
            st.session_state.openai_chat_history = []
        if 'openai_is_typing' not in st.session_state:
            st.session_state.openai_is_typing = False
        
        # API Key Configuration
        with st.form(key="openai_api_form"):
            openai_api_key = st.text_input(
                "OpenAI API Key:", 
                type="password", 
                placeholder="Enter your OpenAI API key...",
                help="Get your API key from https://platform.openai.com/",
                key="openai_key"
            )
            api_key_submitted = st.form_submit_button(" Enter", type="primary")
        
        # Store API key in session state when submitted
        if api_key_submitted and openai_api_key:
            st.session_state.confirmed_openai_key = openai_api_key
            st.success(" OpenAI API Key confirmed!")
        
        # Show API usage info
        if not st.session_state.get('confirmed_openai_key', ''):
            st.info(" **Need an API key?**\n\n"
                   "1. Visit https://platform.openai.com/api-keys\n"
                   "2. Create a new API key\n"
                   "3. Check your billing at https://platform.openai.com/account/billing\n"
                   "4. Make sure you have available quota")
        
        # Use confirmed API key
        confirmed_openai_key = st.session_state.get('confirmed_openai_key', '')
        
        if confirmed_openai_key:
            st.success(f" Using confirmed OpenAI API key: ...{confirmed_openai_key[-4:]}")
        
        if confirmed_openai_key and OPENAI_AVAILABLE:
            render_single_chat("openai", confirmed_openai_key, current_context)
        elif confirmed_openai_key and not OPENAI_AVAILABLE:
            st.error(" OpenAI library is not installed. Please install it with: `pip install openai`")
        else:
            render_welcome_screen("OpenAI GPT-4o", "")
    
    elif ai_mode == "Single Chat (Gemini)":
        st.subheader(" Chat with Google Gemini")
        
        # Initialize chat history for Gemini
        if 'gemini_chat_history' not in st.session_state:
            st.session_state.gemini_chat_history = []
        if 'gemini_is_typing' not in st.session_state:
            st.session_state.gemini_is_typing = False
        
        # API Key Configuration
        with st.form(key="gemini_api_form"):
            gemini_api_key = st.text_input(
                "Gemini API Key:", 
                type="password", 
                placeholder="Enter your Google AI Studio API key...",
                help="Get your API key from https://makersuite.google.com/",
                key="gemini_key"
            )
            api_key_submitted = st.form_submit_button(" Enter", type="primary")
        
        # Store API key in session state when submitted
        if api_key_submitted and gemini_api_key:
            st.session_state.confirmed_gemini_key = gemini_api_key
            st.success(" Gemini API Key confirmed!")
        
        # Show API usage info
        if not st.session_state.get('confirmed_gemini_key', ''):
            st.info(" **Need a Gemini API key?**\n\n"
                   "1. Visit https://makersuite.google.com/app/apikey\n"
                   "2. Create a new API key\n"
                   "3. Gemini has generous free quotas\n"
                   "4. No billing required for basic usage")
        
        # Use confirmed API key
        confirmed_gemini_key = st.session_state.get('confirmed_gemini_key', '')
        
        if confirmed_gemini_key:
            st.success(f" Using confirmed Gemini API key: ...{confirmed_gemini_key[-4:]}")
        
        if confirmed_gemini_key and GEMINI_AVAILABLE:
            render_single_chat("gemini", confirmed_gemini_key, current_context)
        elif confirmed_gemini_key and not GEMINI_AVAILABLE:
            st.error(" Google AI library is not installed. Please install it with: `pip install google-generativeai`")
        else:
            render_welcome_screen("Google Gemini", "")
    
    elif ai_mode == "Compare Models (OpenAI vs Gemini)":
        st.subheader(" Compare AI Models Side by Side")
        
        # Initialize comparison chat histories
        if 'compare_chat_history' not in st.session_state:
            st.session_state.compare_chat_history = []
        if 'compare_is_typing' not in st.session_state:
            st.session_state.compare_is_typing = {"openai": False, "gemini": False}
        
        # API Keys Configuration for Comparison
        with st.form(key="compare_api_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                openai_api_key = st.text_input(
                    "OpenAI API Key:", 
                    type="password", 
                    placeholder="Enter OpenAI API key...",
                    key="compare_openai_key"
                )
            
            with col2:
                gemini_api_key = st.text_input(
                    "Gemini API Key:", 
                    type="password", 
                    placeholder="Enter Gemini API key...",
                    key="compare_gemini_key"
                )
            
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                api_keys_submitted = st.form_submit_button(" Enter Both Keys", type="primary", use_container_width=True)
        
        # Store API keys in session state when submitted
        if api_keys_submitted and openai_api_key and gemini_api_key:
            st.session_state.confirmed_compare_openai_key = openai_api_key
            st.session_state.confirmed_compare_gemini_key = gemini_api_key
            st.success(" Both API Keys confirmed for comparison!")
        elif api_keys_submitted and (not openai_api_key or not gemini_api_key):
            st.error(" Please enter both API keys to enable comparison mode.")
        
        # Use confirmed API keys
        confirmed_openai_key = st.session_state.get('confirmed_compare_openai_key', '')
        confirmed_gemini_key = st.session_state.get('confirmed_compare_gemini_key', '')
        
        if confirmed_openai_key and confirmed_gemini_key:
            col1, col2 = st.columns(2)
            with col1:
                st.success(f" OpenAI: ...{confirmed_openai_key[-4:]}")
            with col2:
                st.success(f" Gemini: ...{confirmed_gemini_key[-4:]}")
        
        if confirmed_openai_key and confirmed_gemini_key and OPENAI_AVAILABLE and GEMINI_AVAILABLE:
            render_comparison_chat(confirmed_openai_key, confirmed_gemini_key, current_context)
        elif not (OPENAI_AVAILABLE and GEMINI_AVAILABLE):
            missing_libs = []
            if not OPENAI_AVAILABLE:
                missing_libs.append("openai")
            if not GEMINI_AVAILABLE:
                missing_libs.append("google-generativeai")
            
            st.error(f" Missing libraries: {', '.join(missing_libs)}. Please install with: `pip install {' '.join(missing_libs)}`")
        else:
            st.info(" Please provide both API keys to start comparison chat!")
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                ###  OpenAI GPT-4o
                - Advanced reasoning capabilities
                - Excellent code understanding
                - Strong financial analysis
                - Creative problem solving
                """)
            
            with col2:
                st.markdown("""
                ###  Google Gemini
                - Multimodal capabilities
                - Fast response times
                - Strong factual knowledge
                - Efficient processing
                """)

def render_single_chat(model_type, api_key, current_context):
    """Render single chat interface"""
    chat_key = f"{model_type}_chat_history"
    typing_key = f"{model_type}_is_typing"
    
    # Clear chat button
    if st.button(f" Clear {model_type.title()} Chat", key=f"clear_{model_type}"):
        st.session_state[chat_key] = []
        st.session_state[typing_key] = False
        st.rerun()
    
    # Chat Messages Display
    if st.session_state[chat_key]:
        for i, message in enumerate(st.session_state[chat_key]):
            if message["role"] == "user":
                st.chat_message("user").write(message["content"])
            else:
                with st.chat_message("assistant"):
                    st.write(message["content"])
                    if "model" in message:
                        st.caption(f" Powered by {message['model']}")
    
    # Show typing indicator
    if st.session_state[typing_key]:
        st.chat_message("assistant").write(f" {model_type.title()} is thinking... ")
    
    # Quick Suggestions (only show if chat is empty)
    if not st.session_state[chat_key]:
        st.markdown("###  Quick Questions")
        suggestions = [
            " Analyze my current stock selection",
            " What are today's market trends?", 
            " Give me investment tips for beginners",
            " Compare tech stocks performance",
            " Explain technical indicators",
            " What are current market risks?"
        ]
        
        cols = st.columns(3)
        for i, suggestion in enumerate(suggestions):
            col = cols[i % 3]
            with col:
                if st.button(suggestion, key=f"{model_type}_suggestion_{i}", use_container_width=True):
                    # Add suggestion as user message
                    question = suggestion.split(" ", 1)[1]
                    st.session_state[chat_key].append({"role": "user", "content": question})
                    st.session_state[typing_key] = True
                    st.rerun()
    
    # Chat Input
    st.markdown("---")
    with st.form(key=f"{model_type}_chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 2])
        
        with col1:
            user_input = st.text_input(
                "Message", 
                placeholder=f"Ask {model_type.title()} about stocks, investments, or market analysis...",
                key=f"{model_type}_chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Send", type="primary", use_container_width=True)
    
    # Handle send message
    if send_button and user_input and user_input.strip():
        # Add user message to history
        st.session_state[chat_key].append({"role": "user", "content": user_input.strip()})
        st.session_state[typing_key] = True
        st.rerun()
    
    # Process AI response if typing
    if st.session_state[typing_key] and st.session_state[chat_key]:
        process_ai_response(model_type, api_key, current_context, chat_key, typing_key)

def render_comparison_chat(openai_api_key, gemini_api_key, current_context):
    """Render comparison chat interface"""
    # Clear comparison chat
    if st.button(" Clear Comparison Chat"):
        st.session_state.compare_chat_history = []
        st.session_state.compare_is_typing = {"openai": False, "gemini": False}
        st.rerun()
    
    # Display chat history
    for message in st.session_state.compare_chat_history:
        if message["role"] == "user":
            st.chat_message("user").write(message["content"])
        else:
            # Display responses side by side
            col1, col2 = st.columns(2)
            
            with col1:
                with st.chat_message("assistant"):
                    st.markdown("** OpenAI GPT-4o**")
                    if "openai_response" in message:
                        st.write(message["openai_response"])
                    else:
                        st.write("Processing...")
            
            with col2:
                with st.chat_message("assistant"):
                    st.markdown("** Google Gemini**") 
                    if "gemini_response" in message:
                        st.write(message["gemini_response"])
                    else:
                        st.write("Processing...")
    
    # Show typing indicators
    if any(st.session_state.compare_is_typing.values()):
        col1, col2 = st.columns(2)
        with col1:
            if st.session_state.compare_is_typing["openai"]:
                st.chat_message("assistant").write(" OpenAI is thinking... ")
        with col2:
            if st.session_state.compare_is_typing["gemini"]:
                st.chat_message("assistant").write(" Gemini is thinking... ")
    
    # Quick suggestions for comparison
    if not st.session_state.compare_chat_history:
        st.markdown("###  Compare AI Responses")
        comparison_questions = [
            "Which stock should I invest in today?",
            "Explain market volatility in simple terms",
            "What's the best investment strategy for 2024?",
            "How do I analyze a company's financial health?",
            "Should I buy cryptocurrency?",
            "Explain the difference between stocks and bonds"
        ]
        
        cols = st.columns(2)
        for i, question in enumerate(comparison_questions):
            col = cols[i % 2]
            with col:
                if st.button(question, key=f"compare_q_{i}", use_container_width=True):
                    start_comparison_response(question, openai_api_key, gemini_api_key, current_context)
    
    # Chat input for comparison
    st.markdown("---")
    with st.form(key="comparison_chat_form", clear_on_submit=True):
        col1, col2 = st.columns([8, 2])
        
        with col1:
            user_input = st.text_input(
                "Message", 
                placeholder="Ask both AI models the same question to compare responses...",
                key="comparison_chat_input",
                label_visibility="collapsed"
            )
        
        with col2:
            send_button = st.form_submit_button("Compare", type="primary", use_container_width=True)
    
    # Handle comparison
    if send_button and user_input and user_input.strip():
        start_comparison_response(user_input.strip(), openai_api_key, gemini_api_key, current_context)
    
    # Process comparison responses if both models are typing
    if any(st.session_state.compare_is_typing.values()) and st.session_state.compare_chat_history:
        process_comparison_responses(openai_api_key, gemini_api_key, current_context)

def start_comparison_response(question, openai_api_key, gemini_api_key, current_context):
    """Start comparison response process"""
    # Add user message
    st.session_state.compare_chat_history.append({"role": "user", "content": question})
    
    # Add placeholder for responses
    st.session_state.compare_chat_history.append({
        "role": "assistant",
        "processing": True
    })
    
    # Set both models as typing
    st.session_state.compare_is_typing = {"openai": True, "gemini": True}
    
    st.rerun()

def process_comparison_responses(openai_api_key, gemini_api_key, current_context):
    """Process responses from both AI models"""
    try:
        user_message = None
        # Find the user message for the current comparison
        for i in range(len(st.session_state.compare_chat_history) - 1, -1, -1):
            if st.session_state.compare_chat_history[i]["role"] == "user":
                user_message = st.session_state.compare_chat_history[i]["content"]
                break
        
        if not user_message:
            return
        
        responses = {}
        
        # Get OpenAI response
        if st.session_state.compare_is_typing["openai"]:
            try:
                responses["openai_response"] = get_openai_response(openai_api_key, user_message, current_context, [])
                st.session_state.compare_is_typing["openai"] = False
            except Exception as e:
                responses["openai_response"] = f"Error: {str(e)}"
                st.session_state.compare_is_typing["openai"] = False
        
        # Get Gemini response
        if st.session_state.compare_is_typing["gemini"]:
            try:
                responses["gemini_response"] = get_gemini_response(gemini_api_key, user_message, current_context, [])
                st.session_state.compare_is_typing["gemini"] = False
            except Exception as e:
                responses["gemini_response"] = f"Error: {str(e)}"
                st.session_state.compare_is_typing["gemini"] = False
        
        # Update the last assistant message with responses
        if responses and st.session_state.compare_chat_history:
            for i in range(len(st.session_state.compare_chat_history) - 1, -1, -1):
                if st.session_state.compare_chat_history[i]["role"] == "assistant" and "processing" in st.session_state.compare_chat_history[i]:
                    st.session_state.compare_chat_history[i].update(responses)
                    if "processing" in st.session_state.compare_chat_history[i]:
                        del st.session_state.compare_chat_history[i]["processing"]
                    break
        
        st.rerun()
        
    except Exception as e:
        st.session_state.compare_is_typing = {"openai": False, "gemini": False}
        st.error(f"Error in comparison: {str(e)}")

def process_ai_response(model_type, api_key, current_context, chat_key, typing_key):
    """Process AI response for single chat"""
    try:
        user_message = st.session_state[chat_key][-1]["content"]
        
        if model_type == "openai":
            response = get_openai_response(api_key, user_message, current_context, st.session_state[chat_key])
            model_name = "OpenAI GPT-4o"
        else:  # gemini
            response = get_gemini_response(api_key, user_message, current_context, st.session_state[chat_key])
            model_name = "Google Gemini"
        
        # Add AI response to history
        st.session_state[chat_key].append({
            "role": "assistant", 
            "content": response,
            "model": model_name
        })
        st.session_state[typing_key] = False
        st.rerun()
        
    except Exception as e:
        st.session_state[typing_key] = False
        
        # Check for specific error types
        error_str = str(e)
        if "429" in error_str or "quota" in error_str.lower():
            error_msg = " **API Quota Exceeded**\n\n"
            error_msg += "Your API quota has been exceeded. Please check:\n"
            error_msg += "• Your OpenAI billing and payment details\n"
            error_msg += "• Your current usage limits\n"
            error_msg += "• Consider upgrading your plan if needed\n\n"
            error_msg += "Visit: https://platform.openai.com/account/billing"
        elif "401" in error_str or "authentication" in error_str.lower():
            error_msg = " **Authentication Error**\n\n"
            error_msg += "Your API key is invalid or expired. Please:\n"
            error_msg += "• Check your API key is correct\n"
            error_msg += "• Ensure it has proper permissions\n"
            error_msg += "• Generate a new key if needed\n\n"
            if model_type == "openai":
                error_msg += "Visit: https://platform.openai.com/api-keys"
            else:
                error_msg += "Visit: https://makersuite.google.com/app/apikey"
        elif "404" in error_str and "model" in error_str.lower():
            error_msg = " **Model Not Available**\n\n"
            if model_type == "gemini":
                error_msg += "The Gemini model is not available. This may be due to:\n"
                error_msg += "• API endpoint changes (try clearing and re-entering your API key)\n"
                error_msg += "• Model availability in your region\n"
                error_msg += "• Try refreshing the page and selecting Gemini again\n\n"
                error_msg += "**Solutions:**\n"
                error_msg += "1. Clear API keys using the button above\n"
                error_msg += "2. Re-enter your Gemini API key\n"
                error_msg += "3. Try switching to OpenAI temporarily\n\n"
                error_msg += "Visit: https://makersuite.google.com/app/apikey"
            else:
                error_msg += f"The {model_type} model is not available. Please check the API documentation."
        elif "rate_limit" in error_str.lower():
            error_msg = " **Rate Limit Exceeded**\n\n"
            error_msg += "Too many requests sent. Please wait a moment and try again.\n"
            error_msg += "Consider spacing out your requests."
        else:
            error_msg = f" **Error**: {error_str}\n\n"
            error_msg += "Please check your API key and try again."
        
        st.session_state[chat_key].append({
            "role": "assistant", 
            "content": error_msg,
            "model": f"{model_type.title()} (Error)"
        })
        st.rerun()

def get_openai_response(api_key, user_message, current_context, chat_history):
    """Get response from OpenAI"""
    try:
        client = OpenAI(api_key=api_key)
        
        system_prompt = f"""You are a knowledgeable financial assistant specialized in stock market analysis.
        
        Current context: {current_context}
        
        Provide helpful, accurate financial insights. Keep responses concise but informative.
        Always include appropriate disclaimers about financial advice.
        Focus on educational content and encourage users to do their own research."""
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history (last 6 messages)
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history[:-1]  # Exclude current message
        for msg in recent_history:
            if "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=400,
            temperature=0.7
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        # Re-raise with more context
        error_msg = str(e)
        if "429" in error_msg:
            raise Exception(f"Error code: 429 - API quota exceeded. {error_msg}")
        elif "401" in error_msg:
            raise Exception(f"Error code: 401 - Authentication failed. {error_msg}")
        else:
            raise Exception(f"OpenAI API Error: {error_msg}")

def get_gemini_response(api_key, user_message, current_context, chat_history):
    """
    Get response from Gemini using Google's recommended approach
    Supports multiple models with proper error handling
    """
    try:
        # Configure API following Google's best practices
        genai.configure(api_key=api_key)
        
        # Updated model names following Google's latest recommendations (2025)
        models_to_try = [
            'gemini-2.5-flash',        # Latest stable fast model (June 2025)
            'gemini-2.5-pro',          # Latest stable pro model (June 2025)
            'gemini-2.0-flash',        # Versatile fast model (Jan 2025)
            'gemini-flash-latest',     # Always latest flash release
            'gemini-pro-latest',       # Always latest pro release
        ]
        
        model = None
        last_error = None
        successful_model = None
        
        # Try each model with proper error handling
        for model_name in models_to_try:
            try:
                model = genai.GenerativeModel(model_name)
                
                # Test the model with a simple prompt to ensure it's working
                # This follows Google's recommendation to validate before using
                test_response = model.generate_content("Test")
                
                if test_response and test_response.text:
                    successful_model = model_name
                    break
                    
            except Exception as e:
                last_error = e
                continue
        
        if model is None:
            error_detail = f"Last error: {last_error}" if last_error else "All models failed"
            raise Exception(f"No available Gemini models found. {error_detail}")
        
        # Build comprehensive context following Google's prompt engineering guidelines
        context_prompt = f"""You are a knowledgeable financial assistant specialized in stock market analysis.

Current Stock Analysis Context: {current_context}

Recent Conversation History:
"""
        
        # Add recent chat history (keep last 6 messages for context)
        recent_history = chat_history[-6:] if len(chat_history) > 6 else chat_history[:-1]
        for msg in recent_history:
            if "content" in msg:
                role = "User" if msg["role"] == "user" else "Assistant"
                context_prompt += f"{role}: {msg['content']}\n"
        
        # Create well-structured prompt following Google's guidelines
        full_prompt = f"""{context_prompt}

Current User Question: {user_message}

Please provide helpful, accurate financial insights following these guidelines:
- Keep responses concise but informative
- Include appropriate disclaimers about financial advice
- Focus on educational content
- Encourage users to do their own research
- Use the current stock analysis context when relevant

Response:"""
        
        # Generate content with error handling for blocked content
        response = model.generate_content(full_prompt)
        
        # Handle potential safety blocks (following Google's documentation)
        try:
            response_text = response.text
            if not response_text:
                raise ValueError("Empty response received")
            return response_text
        except ValueError as ve:
            # Check if content was blocked due to safety filters
            if hasattr(response, 'prompt_feedback'):
                safety_info = f"Content blocked by safety filters: {response.prompt_feedback}"
                return f" Response was filtered for safety reasons. Please try rephrasing your question.\n\nDetails: {safety_info}"
            else:
                raise ve
        
    except Exception as e:
        # Enhanced error handling with specific solutions following Google's recommendations
        error_msg = str(e)
        
        if "404" in error_msg and ("model" in error_msg.lower() or "not found" in error_msg.lower()):
            raise Exception(" **Gemini Model Error**\n\n"
                          "The Gemini API is having issues. Please try:\n"
                          "• **Clear your API keys** using the button above\n"
                          "• **Re-enter your Gemini API key**\n"
                          "• **Try switching to OpenAI** temporarily\n"
                          "• Check if your key works at: https://aistudio.google.com\n\n"
                          f"Technical details: {error_msg}")
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            raise Exception(f" **Gemini API Quota Exceeded**\n\n"
                          f"API usage limit reached: {error_msg}\n\n"
                          f"Try again later or check your quota at Google AI Studio.")
        elif "authentication" in error_msg.lower() or "api_key" in error_msg.lower() or "401" in error_msg:
            raise Exception(f" **Gemini Authentication Failed**\n\n"
                          f"Invalid API key: {error_msg}\n\n"
                          f"Please check your API key at: https://makersuite.google.com/app/apikey")
        else:
            raise Exception(f" **Gemini API Error**\n\n"
                          f"Unexpected error: {error_msg}\n\n"
                          f"Try clearing API keys and re-entering them.")

def render_welcome_screen(model_name, icon):
    """Render welcome screen for AI models"""
    st.markdown(f"""
    <div style='text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 15px; color: white; margin: 2rem 0;'>
        <h3>{icon} Welcome to {model_name}</h3>
        <p>Get personalized stock analysis and investment insights</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ###  Features:
        -  **Smart Analysis**: AI-powered stock insights
        -  **Natural Chat**: Conversation-style interface
        -  **Market Context**: Understands your selections
        -  **Quick Actions**: Pre-built questions
        -  **Educational**: Learn while you invest
        """)
    
    with col2:
        st.markdown("""
        ###  Example Questions:
        - "What's happening with Apple stock?"
        - "Should I buy or sell GOOGL?"
        - "Explain P/E ratios simply"
        - "Market outlook for tech stocks?"
        - "How to diversify my portfolio?"
        """)
    
    st.info(" Please configure your API key above to start chatting!")
    
    # Disclaimer
    st.markdown("""
    <div style='background: #000000; border: 1px solid #333333; border-radius: 8px; padding: 1rem; margin-top: 2rem; color: #ffffff;'>
        <strong> Disclaimer:</strong> AI responses are for educational purposes only. 
        Always conduct your own research and consult with financial professionals before making investment decisions.
    </div>
    """, unsafe_allow_html=True)