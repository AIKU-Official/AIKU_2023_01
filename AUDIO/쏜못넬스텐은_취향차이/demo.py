import gradio as gr
import functions as f
f.map_features()
client_id=input("input client id: ")
client_secret=input("input client secret: ")

f.set_API(client_id,client_secret)

demo = gr.Interface(
    title="쏜못넬스텐은 취향차이",
    description="쏜애플, 못, 넬, 국카스텐의 노래를 추천해드립니다.",
    fn=f.fit_model,
    inputs=gr.inputs.Textbox(lines=5, label="Song URI"),
    outputs=gr.outputs.Textbox(label="Recommendations"),
)

demo.launch(share=True)