import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import base64
from io import BytesIO

# ========================
# Page config
# ========================
st.set_page_config(
    page_title="📦 Package Damage Detection",
    layout="wide"
)

st.title("📦 Package Damage Detection")
st.caption("AI-based package inspection — DAMAGED & INTACT probabilities")

TM_MODEL_URL = "https://teachablemachine.withgoogle.com/models/XPTjPjJOe/"

# ========================
# Tabs
# ========================
tab1, tab2 = st.tabs(["📷 Live Webcam", "📁 Upload Image"])

# ========================
# TAB 1: LIVE WEBCAM - UPDATED TO MATCH UPLOAD TAB
# ========================
with tab1:
    webcam_html = f"""
    <html>
    <head>
        <style>
            * {{
                box-sizing: border-box;
            }}
            body {{
                background: #0f0f0f;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                color: white;
                text-align: center;
                margin: 0;
                padding: 0;
            }}
            .app-container {{
                display: grid;
                grid-template-columns: 1fr 380px;
                grid-template-rows: 1fr;
                gap: 0;
                height: 520px;
                width: 100%;
                max-width: 1200px;
                margin: 0 auto;
                background: #1c1c1c;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
            }}
            .webcam-panel {{
                position: relative;
                background: #111;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}
            #webcam-container {{
                max-width: 100%;
                max-height: 100%;
                border-radius: 12px;
                object-fit: contain;
            }}
            .panel-label {{
                position: absolute;
                bottom: 16px;
                left: 16px;
                background: rgba(0,0,0,0.8);
                color: #fff;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            .controls {{
                position: absolute;
                top: 20px;
                right: 20px;
                display: flex;
                gap: 12px;
            }}
            button {{
                padding: 12px 24px;
                font-size: 14px;
                font-weight: 600;
                border-radius: 12px;
                border: none;
                cursor: pointer;
                transition: all 0.2s ease;
                font-family: inherit;
            }}
            .start-btn {{
                background: #2ecc71;
                color: white;
            }}
            .start-btn:hover {{
                background: #27ae60;
                transform: translateY(-1px);
            }}
            .stop-btn {{
                background: #e74c3c;
                color: white;
            }}
            .stop-btn:hover {{
                background: #c0392b;
                transform: translateY(-1px);
            }}
            .status {{
                position: absolute;
                top: 20px;
                left: 20px;
                background: rgba(0,0,0,0.8);
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            .results-panel {{
                background: #1c1c1c;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .main-result {{
                background: #2d2d2d;
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                border: 1px solid #404040;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .main-title {{
                font-size: 14px;
                color: #aaa;
                font-weight: 500;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .main-prediction {{
                font-size: 36px;
                font-weight: 800;
                margin: 0;
                line-height: 1.1;
            }}
            .confidence {{
                font-size: 20px;
                font-weight: 600;
                margin-top: 4px;
            }}
            .predictions-list {{
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }}
            .prediction-item {{
                padding: 18px 20px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                gap: 16px;
                border: 1px solid #333;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }}
            .prediction-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}
            .damaged-item {{
                background: rgba(231, 76, 60, 0.15);
                border-left: 4px solid #e74c3c;
            }}
            .intact-item {{
                background: rgba(46, 204, 113, 0.15);
                border-left: 4px solid #2ecc71;
            }}
            .pred-label {{
                font-size: 16px;
                font-weight: 700;
                flex: 1;
            }}
            .pred-bar-bg {{
                width: 110px;
                height: 12px;
                background: #333;
                border-radius: 6px;
                overflow: hidden;
                flex-shrink: 0;
            }}
            .pred-bar {{
                height: 100%;
                border-radius: 6px;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            .damaged-bar {{
                background: linear-gradient(90deg, #e74c3c, #ff6b7a);
            }}
            .intact-bar {{
                background: linear-gradient(90deg, #2ecc71, #27ae60);
            }}
            .pred-percent {{
                font-size: 18px;
                font-weight: 800;
                min-width: 50px;
                text-align: right;
            }}
        </style>
    </head>
    <body>
        <div class="app-container">
            <!-- WEBCAM PANEL -->
            <div class="webcam-panel">
                <div id="webcam-container"></div>
                <div class="panel-label">📷 Live Webcam</div>
                <div class="controls">
                    <button class="start-btn" onclick="startWebcam()">▶ Start</button>
                    <button class="stop-btn" onclick="stopWebcam()">⏹ Stop</button>
                </div>
                <div id="status" class="status">Status: Idle</div>
            </div>
            
            <!-- RESULTS PANEL -->
            <div class="results-panel">
                <div class="main-result">
                    <div class="main-title">Live Detection</div>
                    <div id="main-prediction" class="main-prediction">Waiting...</div>
                    <div id="confidence" class="confidence">00%</div>
                </div>
                
                <div class="predictions-list">
                    <div class="prediction-item damaged-item">
                        <div class="pred-label">DAMAGED</div>
                        <div class="pred-bar-bg">
                            <div id="damaged-bar" class="pred-bar damaged-bar" style="width: 0%"></div>
                        </div>
                        <div id="damaged-text" class="pred-percent">0%</div>
                    </div>
                    <div class="prediction-item intact-item">
                        <div class="pred-label">INTACT</div>
                        <div class="pred-bar-bg">
                            <div id="intact-bar" class="pred-bar intact-bar" style="width: 0%"></div>
                        </div>
                        <div id="intact-text" class="pred-percent">0%</div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>

        <script>
            const URL="{TM_MODEL_URL}";
            let model, webcam, running = false;

            async function startWebcam() {{
                if(running) return;
                
                document.getElementById("status").textContent = "Status: Loading model...";
                try {{
                    model = await tmImage.load(URL+"model.json", URL+"metadata.json");
                    
                    webcam = new tmImage.Webcam(480, 480, true);
                    await webcam.setup();
                    await webcam.play();
                    
                    running = true;
                    document.getElementById("status").textContent = "Status: Live";
                    const container = document.getElementById("webcam-container");
                    container.innerHTML = "";
                    container.appendChild(webcam.canvas);
                    
                    loop();
                }} catch (error) {{
                    console.error("Webcam error:", error);
                    document.getElementById("status").textContent = "Status: Error";
                }}
            }}

            async function loop() {{
                if(!running) return;
                webcam.update();
                await predict();
                requestAnimationFrame(loop);
            }}

            async function predict() {{
                try {{
                    const predictions = await model.predict(webcam.canvas);
                    let damaged = 0, intact = 0;
                    
                    predictions.forEach(p => {{
                        if(p.className.toLowerCase().includes("damage")) {{
                            damaged = Math.max(damaged, p.probability);
                        }} else {{
                            intact = Math.max(intact, p.probability);
                        }}
                    }});

                    const damagedPct = Math.round(damaged * 100);
                    const intactPct = Math.round(intact * 100);
                    
                    const mainClass = damagedPct > intactPct ? 'DAMAGED' : 'INTACT';
                    const mainConf = Math.max(damagedPct, intactPct);
                    const mainColor = mainClass === 'DAMAGED' ? '#e74c3c' : '#2ecc71';
                    
                    document.getElementById('main-prediction').textContent = mainClass;
                    document.getElementById('main-prediction').style.color = mainColor;
                    document.getElementById('confidence').textContent = mainConf + '%';
                    document.getElementById('confidence').style.color = mainColor;
                    
                    document.getElementById('damaged-bar').style.width = damagedPct + '%';
                    document.getElementById('damaged-text').textContent = damagedPct + '%';
                    document.getElementById('damaged-text').style.color = '#e74c3c';
                    
                    document.getElementById('intact-bar').style.width = intactPct + '%';
                    document.getElementById('intact-text').textContent = intactPct + '%';
                    document.getElementById('intact-text').style.color = '#2ecc71';
                    
                }} catch (error) {{
                    console.error('Prediction error:', error);
                }}
            }}

            function stopWebcam() {{
                if(!running) return;
                running = false;
                if(webcam) webcam.stop();
                document.getElementById("status").textContent = "Status: Stopped";
                document.getElementById("webcam-container").innerHTML = "";
                // Reset all values
                document.getElementById('main-prediction').textContent = 'Waiting...';
                document.getElementById('confidence').textContent = '00%';
                document.getElementById('damaged-bar').style.width = '0%';
                document.getElementById('damaged-text').textContent = '0%';
                document.getElementById('intact-bar').style.width = '0%';
                document.getElementById('intact-text').textContent = '0%';
            }}
        </script>
    </body>
    </html>
    """
    components.html(webcam_html, height=560)

# ========================
# TAB 2: UPLOAD IMAGE (same design as webcam)
# ========================
with tab2:
    uploaded_file = st.file_uploader("Upload Image", type=["jpg","jpeg","png"], key="file_uploader")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()
        
        upload_html = f"""
        <style>
            /* SAME EXACT STYLES AS WEBCAM TAB */
            * {{ box-sizing: border-box; }}
            .app-container {{
                display: grid;
                grid-template-columns: 1fr 380px;
                grid-template-rows: 1fr;
                gap: 0;
                height: 500px;
                width: 100%;
                background: #1c1c1c;
                border-radius: 16px;
                overflow: hidden;
                box-shadow: 0 6px 20px rgba(0, 0, 0, 0.5);
            }}
            .image-panel {{
                position: relative;
                background: #111;
                display: flex;
                align-items: center;
                justify-content: center;
                overflow: hidden;
            }}
            .image-panel img {{
                max-width: 100%;
                max-height: 100%;
                object-fit: contain;
            }}
            .image-label {{
                position: absolute;
                bottom: 16px;
                left: 16px;
                background: rgba(0,0,0,0.8);
                color: #fff;
                padding: 8px 16px;
                border-radius: 20px;
                font-size: 13px;
                font-weight: 500;
            }}
            .results-panel {{
                background: #1c1c1c;
                padding: 24px;
                display: flex;
                flex-direction: column;
                gap: 20px;
            }}
            .main-result {{
                background: #2d2d2d;
                padding: 24px;
                border-radius: 16px;
                text-align: center;
                border: 1px solid #404040;
                box-shadow: 0 4px 12px rgba(0,0,0,0.2);
            }}
            .main-title {{
                font-size: 14px;
                color: #aaa;
                font-weight: 500;
                margin-bottom: 12px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }}
            .main-prediction {{
                font-size: 36px;
                font-weight: 800;
                margin: 0;
                line-height: 1.1;
            }}
            .confidence {{
                font-size: 20px;
                font-weight: 600;
                margin-top: 4px;
            }}
            .predictions-list {{
                flex: 1;
                display: flex;
                flex-direction: column;
                gap: 16px;
            }}
            .prediction-item {{
                padding: 18px 20px;
                border-radius: 12px;
                display: flex;
                align-items: center;
                gap: 16px;
                border: 1px solid #333;
                transition: all 0.3s ease;
            }}
            .prediction-item:hover {{
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(0,0,0,0.3);
            }}
            .damaged-item {{
                background: rgba(231, 76, 60, 0.15);
                border-left: 4px solid #e74c3c;
            }}
            .intact-item {{
                background: rgba(46, 204, 113, 0.15);
                border-left: 4px solid #2ecc71;
            }}
            .pred-label {{
                font-size: 16px;
                font-weight: 700;
                flex: 1;
            }}
            .pred-bar-bg {{
                width: 110px;
                height: 12px;
                background: #333;
                border-radius: 6px;
                overflow: hidden;
                flex-shrink: 0;
            }}
            .pred-bar {{
                height: 100%;
                border-radius: 6px;
                transition: width 0.8s cubic-bezier(0.4, 0, 0.2, 1);
            }}
            .damaged-bar {{
                background: linear-gradient(90deg, #e74c3c, #ff6b7a);
            }}
            .intact-bar {{
                background: linear-gradient(90deg, #2ecc71, #27ae60);
            }}
            .pred-percent {{
                font-size: 18px;
                font-weight: 800;
                min-width: 50px;
                text-align: right;
            }}
        </style>

        <div class="app-container">
            <div class="image-panel">
                <img src="data:image/png;base64,{img_base64}">
                <div class="image-label">📁 Uploaded Package</div>
            </div>
            <div class="results-panel">
                <div class="main-result">
                    <div class="main-title">Prediction Result</div>
                    <div id="main-prediction" class="main-prediction">Analyzing...</div>
                    <div id="confidence" class="confidence">Loading...</div>
                </div>
                <div class="predictions-list">
                    <div class="prediction-item damaged-item">
                        <div class="pred-label">DAMAGED</div>
                        <div class="pred-bar-bg">
                            <div id="damaged-bar" class="pred-bar damaged-bar" style="width: 0%"></div>
                        </div>
                        <div id="damaged-text" class="pred-percent">0%</div>
                    </div>
                    <div class="prediction-item intact-item">
                        <div class="pred-label">INTACT</div>
                        <div class="pred-bar-bg">
                            <div id="intact-bar" class="pred-bar intact-bar" style="width: 0%"></div>
                        </div>
                        <div id="intact-text" class="pred-percent">0%</div>
                    </div>
                </div>
            </div>
        </div>

        <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest/dist/tf.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/@teachablemachine/image@latest/dist/teachablemachine-image.min.js"></script>
        <script>
            const URL="{TM_MODEL_URL}";
            async function predictImage() {{
                try {{
                    const model = await tmImage.load(URL+"model.json", URL+"metadata.json");
                    const img = new Image();
                    img.src="data:image/png;base64,{img_base64}";
                    await new Promise(resolve => {{ img.onload = resolve; }});
                    const predictions = await model.predict(img);
                    let damaged = 0, intact = 0;
                    predictions.forEach(p => {{
                        if(p.className.toLowerCase().includes("damage")) {{
                            damaged = Math.max(damaged, p.probability);
                        }} else {{
                            intact = Math.max(intact, p.probability);
                        }}
                    }});
                    const damagedPct = Math.round(damaged * 100);
                    const intactPct = Math.round(intact * 100);
                    const mainClass = damagedPct > intactPct ? 'DAMAGED' : 'INTACT';
                    const mainConf = Math.max(damagedPct, intactPct);
                    const mainColor = mainClass === 'DAMAGED' ? '#e74c3c' : '#2ecc71';
                    document.getElementById('main-prediction').textContent = mainClass;
                    document.getElementById('main-prediction').style.color = mainColor;
                    document.getElementById('confidence').textContent = mainConf + '%';
                    document.getElementById('confidence').style.color = mainColor;
                    document.getElementById('damaged-bar').style.width = damagedPct + '%';
                    document.getElementById('damaged-text').textContent = damagedPct + '%';
                    document.getElementById('damaged-text').style.color = '#e74c3c';
                    document.getElementById('intact-bar').style.width = intactPct + '%';
                    document.getElementById('intact-text').textContent = intactPct + '%';
                    document.getElementById('intact-text').style.color = '#2ecc71';
                }} catch (error) {{
                    console.error('Prediction error:', error);
                }}
            }}
            predictImage();
        </script>
        """
        components.html(upload_html, height=520)

    else:
        st.info("👆 Upload an image to see prediction results")

# Footer
st.markdown("---")
st.caption("Academic demo • Binary classification: DAMAGED vs INTACT")
