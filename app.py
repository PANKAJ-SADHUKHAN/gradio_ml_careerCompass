import pickle
import gradio as gr
import numpy as np
import joblib
#gr.themes.builder()
theme = gr.themes.Soft(
    primary_hue="violet",
    secondary_hue="slate",
    font=['apple', 'ui-sans-serif', 'system-ui', 'sans-serif'],
)


# Step 1: Load the model
model=joblib.load('model\data_model.pkl')
job_titles = {
    0: 'AI ML Specialist',
    1: 'API Specialist',
    2: 'Application Support Engineer',
    3: 'Business Analyst',
    4: 'Customer Service Executive',
    5: 'Cyber Security Specialist',
    6: 'Database Administrator',
    7: 'Graphics Designer',
    8: 'Hardware Engineer',
    9: 'Helpdesk Engineer',
    10: 'Information Security Specialist',
    11: 'Networking Engineer',
    12: 'Project Manager',
    13: 'Software Developer',
    14: 'Software Tester',
    15: 'Technical Writer'
}
# Step 2: Define the prediction function
def predict(discrete1, discrete2, discrete3, discrete4, continuous5, continuous6, continuous7, continuous8, continuous9, continuous10, continuous11,continuous12,continuous13,continuous14):
    input_data = np.array([discrete1, discrete2, discrete3, discrete4, continuous5, continuous6, continuous7, continuous8, continuous9, continuous10, continuous11,continuous12,continuous13,continuous14]).reshape(1,-1)
    print(f"Input data format: {input_data}")  # Print the input data for verification
    prediction = model.predict(input_data)
    predicted_job_title = job_titles[prediction[0]]
    
    return predicted_job_title 
# Step 3: Create the Gradio interface
with gr.Blocks(theme=theme) as demo:
    input_components = [
        gr.Slider(minimum=1, maximum=6, step=1, label="Computer Architecture"),
        gr.Slider(minimum=1, maximum=6, step=1, label="Programming skill"),
        gr.Slider(minimum=1, maximum=6, step=1, label="Project Management"),
        gr.Slider(minimum=1, maximum=6, step=1, label="Communication Skill"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Openness"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Conscientiousness"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Extraversion"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Agreebleness"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Emotionalness"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Conversation"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Openness to change"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Hedonism"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Self Enhancement"),
        gr.Slider(minimum=0, maximum=1, step=0.01, label="Self Transcendence"),
]

    output_component = gr.Textbox(label="Prediction")


    iface = gr.Interface(
        fn=predict,
        inputs=input_components,
        outputs=output_component,
        title="Model Predictor",
        description="Enter the values for the inputs to get a prediction.",
)

# Launch the interface
#iface.launch(share=True)
demo.launch(share=True)