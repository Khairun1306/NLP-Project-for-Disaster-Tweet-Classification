import gradio as gr
from predictor import predict_tweet

examples = [
    ["Our Deeds are the Reason of this #earthquake May ALLAH Forgive us all"],
    ["Massive earthquake destroys several buildings in Turkey"],
    ["Flood warning issued for coastal areas. Residents should evacuate immediately."],
    ["I love spending time at the beach with my family."],
    ["Watching Netflix tonight with friends."],
    ["Wildfire near Los Angeles forces thousands to evacuate."]
]

with gr.Blocks(
    title="🌪️ Disaster Tweet Classification",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown("""
# 🌪️ Disaster Tweet Classification

Classify tweets as **Disaster** or **Non-Disaster** using a Machine Learning model.

**Model:** Logistic Regression (Best Model)
""")

    with gr.Row():

        with gr.Column(scale=2):

            tweet = gr.Textbox(
                label="📝 Tweet",
                placeholder="Type or paste a tweet here...",
                lines=4
            )

            with gr.Row():
                predict_btn = gr.Button("🔍 Classify", variant="primary")
                clear_btn = gr.ClearButton([tweet])

        with gr.Column(scale=1):

            prediction = gr.Textbox(
                label="Prediction",
                interactive=False
            )

            confidence = gr.Textbox(
                label="Confidence",
                interactive=False
            )

    gr.Examples(
        examples=examples,
        inputs=tweet
    )

    predict_btn.click(
        fn=predict_tweet,
        inputs=tweet,
        outputs=[prediction, confidence]
    )

    gr.Markdown(
        """
---
Built with **Scikit-learn**, **NLTK**, **Gradio**, and **XGBoost**
"""
    )

if __name__ == "__main__":
    demo.launch()