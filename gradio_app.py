import gradio as gr

def assess_pcos_risk(handwashing, sanitation, pm25, gdp):
    # Calculate risk based on a simplified formula
    risk_score = (100 - handwashing) * 0.1 + \
                 (100 - sanitation) * 0.1 + \
                 (pm25 / 50) * 10 + \
                 (25 / (gdp + 1)) * 10
    
    risk_score = max(0, min(100, risk_score))

    if risk_score < 30:
        level = "🟢 Low Risk"
    elif risk_score < 60:
        level = "🟡 Moderate Risk"
    else:
        level = "🔴 High Risk"
    
    result = f"**Risk Score:** {risk_score:.1f} / 100\n\n**Risk Level:** {level}"
    return result

# Create the Gradio interface
iface = gr.Interface(
    fn=assess_pcos_risk,
    inputs=[
        gr.Slider(0, 100, value=85, label="Handwashing Facilities Coverage (%)"),
        gr.Slider(0, 100, value=90, label="Safe Sanitation Facilities (%)"),
        gr.Slider(0, 50, value=15, label="PM2.5 Exposure (μg/m³)"),
        gr.Slider(1, 100, value=25, label="GDP per Capita (in thousands USD)")
    ],
    outputs=gr.Markdown(label="Assessment Results"),
    title="🔬 PCOS Environmental Health Risk Assessment Tool",
    description="A simplified tool to assess PCOS risk based on environmental factors. Based on GBD research.",
    theme=gr.themes.Soft()
)

# Launch the app
if __name__ == "__main__":
    iface.launch()