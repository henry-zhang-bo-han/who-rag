import streamlit as st
from openai import OpenAI

if __name__ == '__main__':
    st.title('WHO Document Search')

    # Initiate OpenAI instance and vector stores
    if 'openai' not in st.session_state:
        st.session_state['openai'] = OpenAI()

    if 'vector_store' not in st.session_state:
        st.session_state['vector_store'] = st.session_state['openai'].beta.vector_stores.create(name='WHO')
        st.session_state['uploaded_file_names'] = set()

    if 'assistant' not in st.session_state:
        st.session_state['assistant'] = st.session_state['openai'].beta.assistants.create(
            name='WHO Assistant',
            instructions='You are an expert health analyst. Use your knowledge base to answer questions from the user.',
            model='gpt-4o-mini',
            tools=[{'type': 'file_search'}],
            tool_resources={'file_search': {'vector_store_ids': [st.session_state['vector_store'].id]}}
        )

    # Allow user to upload one or multiple files
    with st.sidebar:
        uploaded_files = st.file_uploader(
            'Upload file(s)',
            type=['pdf', 'docx'],
            accept_multiple_files=True
        )

    # Process newly uploaded files
    file_streams = [f for f in uploaded_files if f.name not in st.session_state['uploaded_file_names']]
    if len(file_streams) > 0:
        st.session_state['uploaded_file_names'].update({f.name for f in uploaded_files})
        file_batch = st.session_state['openai'].beta.vector_stores.file_batches.upload_and_poll(
            vector_store_id=st.session_state['vector_store'].id,
            files=file_streams
        )

    # Initialize messages for chatbot
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask AI questions about WHO ..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)

        # Create thread and run for assistant
        client = st.session_state['openai']
        thread = client.beta.threads.create(messages=st.session_state['messages'])
        run = client.beta.threads.runs.create_and_poll(
            thread_id=thread.id,
            assistant_id=st.session_state['assistant'].id
        )
        messages = list(client.beta.threads.messages.list(thread_id=thread.id, run_id=run.id))

        with st.chat_message('assistant'):
            for message in messages:
                for content in message.content:
                    # Process annotations
                    message_content = content.text
                    annotations = message_content.annotations
                    citations = []
                    for idx, annotation in enumerate(annotations):
                        message_content.value = message_content.value.replace(annotation.text, f'[{idx + 1}]')
                        if file_citation := getattr(annotation, "file_citation", None):
                            cited_file = st.session_state['openai'].files.retrieve(file_citation.file_id)
                            citations.append(f" [{idx + 1}] {cited_file.filename}")

                    # Format citations
                    citation_string = '\n\n**Citations:**\n\n' + '\n\n'.join(citations)
                    message_content.value += citation_string
                    message_content.value = message_content.value.replace('$', '\\$')

                    # Display message in chat message container
                    st.markdown(message_content.value)
                    st.session_state['messages'].append(
                        {"role": "assistant", "content": message_content.value}
                    )
