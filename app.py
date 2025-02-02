import gradio as gr
from langchain_core.messages import HumanMessage
from create_graph import graph

class ChatInterface:
    def __init__(self):
        self.config = {"configurable": {"thread_id": "1"}}
        self.chat_history = []
    
    def chat(self, message, history):
        """Process chat messages and maintain history."""
        if not message:
            return "", history
        
        try:
            final_state = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=self.config
            )
            
            bot_response = final_state["messages"][-1].content
            
            history.append((message, bot_response))
            
            return "", history
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            history.append((message, error_message))
            return "", history

    def create_interface(self):
        """Create and configure the Gradio interface."""
        # Define chat interface
        chat_interface = gr.Blocks(theme=gr.themes.Soft())
        
        with chat_interface:
            gr.Markdown("#JurisAI Chatbot")
            gr.Markdown("Welcome to our JurisAI chatbot! Ask me anything that's legal.")
            
            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                container=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Type your message here...",
                    container=False
                )
                submit = gr.Button("Send", scale=1)
            
            clear = gr.Button("Clear Chat")
            
            submit_click = submit.click(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            msg_submit = msg.submit(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)
            
        return chat_interface

def main():
    chat_app = ChatInterface()
    
    chat_app.create_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()