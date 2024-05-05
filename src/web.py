import gradio as gr
import time

def slow_echo(message, history):
    if len(history) % 2 == 0:
        yield f"Yes, I do think that '{message}'"
    else:
        for i in range(len(message)):
            time.sleep(0.3)
            yield "You typed: " + message[: i+1]

gr.ChatInterface(slow_echo,
                 textbox=gr.Textbox(placeholder="Ask me a yes or no question", container=False, scale=7),
                 title="Yes Man",
                 description="Ask Yes Man any question",
                 theme="soft",
                 # examples=["Hello", "Am I cool?", "Are tomatoes vegetables?"],
                 cache_examples=True,
                 retry_btn=None,
                 undo_btn="Delete Previous",
                 clear_btn="Clear",
                 ).launch()