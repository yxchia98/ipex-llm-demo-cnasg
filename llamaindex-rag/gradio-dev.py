import gradio as gr
import time
from tqdm import tqdm


css = """
.app-interface {
    height:90vh;
}
.chat-interface {
    height: 90vh;
}
.file-interface {
    height: 40vh;
}
.web-interface {
    height: 30vh;
}
"""


def stream_response(message, history):
    for i in range(len(message)):
        time.sleep(0.05)
        yield "You typed: " + message[: i + 1]

def vectorize(files, progress=gr.Progress()):
    print('vectorize!')
    progress(0, desc="Starting")
    time.sleep(1)
    progress(0.05)
    new_string = ""
    file_paths = [file.name for file in files]
    for file in progress.tqdm(files, desc="Vectorizing..."):
        print(file)
        time.sleep(1)
    return file_paths




with gr.Blocks(css=css) as demo:
    with gr.Row(equal_height=True, elem_classes=["app-interface"]):
        with gr.Column(scale=4, elem_classes=["chat-interface"]):
            test = gr.ChatInterface(stream_response)
        with gr.Column(scale=1):
            file_input = gr.File(elem_classes=["file-interface"], file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button = gr.UploadButton("Click to Upload a File", file_types=["image", "video", "pdf", "csv", "text"], file_count="multiple")
            # upload_button.upload(upload_file, upload_button, file_input)
            vectorize_button = gr.Button("Vectorize Files")
            vectorize_button.click(fn=vectorize, inputs=file_input, outputs=file_input)
            

demo.launch()
