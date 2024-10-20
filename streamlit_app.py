import streamlit as st
#import dotenv
from toolhouse import Toolhouse
from toolhouse_llamaindex import ToolhouseLlamaIndex
from together import Together
import json

# Load environment variables
#dotenv.load_dotenv()

# Streamlit app title
st.title("FizzBuzz Generator with Toolhouse and Together AI")

# Initialize Toolhouse
TOOLHOUSE_API_KEY=st.secrets["TOOLHOUSE_API_KEY"]
th = Toolhouse(api_key=TOOLHOUSE_API_KEY)
th.set_metadata("id", "10566")
th.set_metadata("timezone", -8)
th.bundle = "pinecone"  # optional, only if you want to use bundles

ToolhouseSpec = ToolhouseLlamaIndex(th)
tool_spec = ToolhouseSpec()

# Initialize Together AI client
TOGETHER_API_KEY = st.secrets["TOGETHER_API_KEY"]
client = Together(api_key=TOGETHER_API_KEY)
MODEL = 'mistralai/Mixtral-8x7B-Instruct-v0.1'

# Initial message
messages = [{
    "role": "user",
    "content": "Generate FizzBuzz code. Execute it to show me the results up to 10."
}]

# Function to make API call and handle errors
def make_api_call(messages):
    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=th.get_tools(),
        )
        return response
    except Exception as e:
        st.error(f"An error occurred during API call: {str(e)}")
        return None

# Make the first API call
response = make_api_call(messages)

if response:
    st.write("Initial response received. Processing...")

    # Display the raw response for debugging
    st.subheader("Raw API Response:")
    st.json(response.model_dump())

    # Run tools and handle potential errors
    try:
        tool_run = th.run_tools(response)
        messages.append(tool_run)
        st.write("Tool execution successful. Making final API call...")
    except KeyError as e:
        st.error(f"KeyError occurred during tool execution: {str(e)}")
        st.write("This error suggests that the 'function_call' key is missing from the response.")
        st.write("Let's check the structure of the 'choices' in the response:")

        for i, choice in enumerate(response.choices):
            st.subheader(f"Choice {i}:")
            st.json(choice.model_dump())

            # Check if 'message' exists and has a 'function_call' key
            if hasattr(choice, 'message'):
                if hasattr(choice.message, 'function_call'):
                    st.write("Function call found in this choice:")
                    st.json(choice.message.function_call)
                else:
                    st.write("No 'function_call' found in this choice's message.")
            else:
                st.write("No 'message' attribute found in this choice.")

    except Exception as e:
        st.error(f"An error occurred during tool execution: {str(e)}")

    # Make the final API call
    final_response = make_api_call(messages)

    if final_response:
        st.write("Final Response:")
        st.write(final_response.choices[0].message.content)
else:
    st.error("Failed to get initial response from the API.")

# Display the full conversation for debugging
st.write("Full Conversation:")
for msg in messages:
    st.write(f"{msg['role']}: {msg['content']}")
