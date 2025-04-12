import gradio as gr

from sentence_transformers import SentenceTransformer, util
from pregnancy_data import pregnancy_qa_data # import dataset

# load pre-trained sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2') # good balance of speed and accuracy

# prepare questions and answers for embedding
questions = [item['question'] for item in pregnancy_qa_data]
answers = [item['answer'] for item in pregnancy_qa_data]

# embed all questions in your dataset
question_embeddings = model.encode(questions, convert_to_tensor=True)

def get_chatbot_response(user_question):
    """
    Finds the most similar question in the dataset to the user's question
    and returns the corresponding answer.
    """
    user_question_embedding = model.encode(user_question, convert_to_tensor=True)

    # calculate cosine similarity between user question and all questions in the dataset
    cosine_scores = util.pytorch_cos_sim(user_question_embedding, question_embeddings)[0]

    # find the index of the most similar question
    most_similar_index = cosine_scores.argmax()

    # return the answer corresponding to the most similar question
    return answers[most_similar_index]

def chatbot_interface(user_input):
    return get_chatbot_response(user_input)

iface = gr.Interface(
    fn=gr.Interface,
        inputs=[
            gr.Textbox(placeholder="Ask a pregnancy question...", label="Your Question"),
            gr.State([]) # to keep track of conversation history
        ],
    outputs=gr.Chatbot(label="Chat Room"),
    title="Afya Mkononi Chatbot (Prototype)",
    description="This is a simple prototype chatbot for answering pregnancy-related questions." \
    "**Disclaimer: This is for demonstration purposes only and is NOT a substitute for professional medical advice. The information provided is not medically validated.**"
)
iface.launch()

# Example usage
if __name__ == "__main__":
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break
        response = get_chatbot_response(user_input)
        print("Chatbot:", response)