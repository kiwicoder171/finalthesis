#!/usr/bin/env python3
"""
generate_pdf.py
A pure-Python dependency-free PDF 1.4 generator for the Capstone Thesis Proposal Report.
Generates an elegant, publication-quality multi-page PDF document.
"""

import os
import sys

class PDFBuilder:
    def __init__(self, page_width=595.28, page_height=841.89): # A4
        self.w = page_width
        self.h = page_height
        self.pages = []
        self.current_page_ops = []
        self.objects = []
        
    def new_page(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []
        self.current_page_ops = []
        # Page header & footer setup
        self._add_page_decorations()

    def _add_page_decorations(self):
        page_num = len(self.pages) + 1
        # Top rule
        self.set_stroke_color(0.8, 0.85, 0.9)
        self.set_line_width(0.8)
        self.line(45, self.h - 45, self.w - 45, self.h - 45)
        
        # Header text
        self.draw_text("COMPUTER ENGINEERING CAPSTONE THESIS PROPOSAL", 45, self.h - 38, 7.5, "Courier", (0.4, 0.45, 0.5))
        self.draw_text("AUTONOMIC CARDIAC DIGITAL TWIN", self.w - 200, self.h - 38, 7.5, "Courier", (0.4, 0.45, 0.5))
        
        # Bottom rule
        self.line(45, 45, self.w - 45, 45)
        self.draw_text("CONFIDENTIAL • ACADEMIC USE ONLY", 45, 34, 7.5, "Courier", (0.4, 0.45, 0.5))
        self.draw_text(f"Page {page_num}", self.w - 80, 34, 7.5, "Courier", (0.4, 0.45, 0.5))

    def set_fill_color(self, r, g, b):
        self.current_page_ops.append(f"{r:.3f} {g:.3f} {b:.3f} rg")

    def set_stroke_color(self, r, g, b):
        self.current_page_ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG")

    def set_line_width(self, w):
        self.current_page_ops.append(f"{w:.2f} w")

    def rect(self, x, y, width, height, fill=True, stroke=False):
        op = "B" if fill and stroke else ("f" if fill else "S")
        self.current_page_ops.append(f"{x:.2f} {y:.2f} {width:.2f} {height:.2f} re {op}")

    def line(self, x1, y1, x2, y2):
        self.current_page_ops.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def escape_text(self, text):
        replacements = {
            '•': '-',
            '–': '-',
            '—': '--',
            '’': "'",
            '‘': "'",
            '“': '"',
            '”': '"',
            '…': '...',
            '°': ' deg',
            'µ': 'u',
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        # Ensure latin1 safe
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    def draw_text(self, text, x, y, size=10, font="Helvetica", color=(0.1, 0.1, 0.15)):
        r, g, b = color
        escaped = self.escape_text(text)
        ops = [
            "BT",
            f"/{font} {size:.2f} Tf",
            f"{r:.3f} {g:.3f} {b:.3f} rg",
            f"1 0 0 1 {x:.2f} {y:.2f} Tm",
            f"({escaped}) Tj",
            "ET"
        ]
        self.current_page_ops.append(" ".join(ops))

    def finalize_page(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []

    def build_pdf(self, output_path):
        self.finalize_page()
        
        # We assemble objects
        # 1: Catalog
        # 2: Outlines
        # 3: Pages
        # 4..: Font objects
        # Then page objects & content streams
        
        objects = []
        
        # Placeholder for offsets
        def add_obj(body):
            objects.append(body)
            return len(objects)

        cat_id = add_obj("<< /Type /Catalog /Pages 3 0 R >>")
        outlines_id = add_obj("<< /Type /Outlines /Count 0 >>")
        
        # Pages object is 3
        # We will patch pages object later
        pages_id = add_obj("") # will be replaced
        
        # Fonts
        font_helv = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        font_helv_b = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        font_helv_o = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")
        font_cour = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
        font_cour_b = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold >>")
        font_times = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Roman >>")
        font_times_b = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Times-Bold >>")

        page_refs = []
        for page_ops in self.pages:
            content_stream = "\n".join(page_ops)
            content_bytes = content_stream.encode('latin1')
            content_id = add_obj(f"<< /Length {len(content_bytes)} >>\nstream\n{content_stream}\nendstream")
            
            page_obj = (
                f"<< /Type /Page\n"
                f"/Parent {pages_id} 0 R\n"
                f"/MediaBox [0 0 {self.w:.2f} {self.h:.2f}]\n"
                f"/Contents {content_id} 0 R\n"
                f"/Resources << /Font <<\n"
                f"  /Helvetica {font_helv} 0 R\n"
                f"  /Helvetica-Bold {font_helv_b} 0 R\n"
                f"  /Helvetica-Oblique {font_helv_o} 0 R\n"
                f"  /Courier {font_cour} 0 R\n"
                f"  /Courier-Bold {font_cour_b} 0 R\n"
                f"  /Times-Roman {font_times} 0 R\n"
                f"  /Times-Bold {font_times_b} 0 R\n"
                f">> >> >>"
            )
            p_id = add_obj(page_obj)
            page_refs.append(f"{p_id} 0 R")

        # Now patch Pages object
        kids_str = " ".join(page_refs)
        objects[pages_id - 1] = f"<< /Type /Pages /Kids [ {kids_str} ] /Count {len(page_refs)} >>"

        # Now output PDF file with XRef table
        with open(output_path, "wb") as f:
            f.write(b"%PDF-1.4\n")
            xref_offsets = []
            
            for i, obj in enumerate(objects):
                offset = f.tell()
                xref_offsets.append(offset)
                obj_str = f"{i+1} 0 obj\n{obj}\nendobj\n"
                f.write(obj_str.encode('latin1'))
                
            xref_pos = f.tell()
            f.write(f"xref\n0 {len(objects) + 1}\n".encode('latin1'))
            f.write(b"0000000000 65535 f \n")
            for off in xref_offsets:
                f.write(f"{off:010d} 00000 n \n".encode('latin1'))
                
            trailer = (
                f"trailer\n"
                f"<< /Size {len(objects) + 1}\n"
                f"/Root {cat_id} 0 R >>\n"
                f"startxref\n{xref_pos}\n%%EOF\n"
            )
            f.write(trailer.encode('latin1'))
            
        print(f"Successfully generated {output_path} ({len(self.pages)} pages)")

def generate_thesis_proposal():
    doc = PDFBuilder()
    
    # ==========================================
    # PAGE 1: TITLE, ABSTRACT & ARCHITECTURE
    # ==========================================
    doc.new_page()
    y = doc.h - 75
    
    # Title Tag
    doc.set_fill_color(0.92, 0.35, 0.1) # Coral Accent
    doc.rect(45, y, 75, 14, fill=True, stroke=False)
    doc.draw_text("CAPSTONE PROPOSAL", 49, y + 3.5, 7, "Courier-Bold", (1, 1, 1))
    
    y -= 26
    # Title
    doc.draw_text("A Dual-Modal Mobile Sensing System for Real-Time Cardiac", 45, y, 16, "Helvetica-Bold", (0.06, 0.09, 0.16))
    y -= 19
    doc.draw_text("Autonomic Risk Profiling and Dynamical Stress Simulation", 45, y, 16, "Helvetica-Bold", (0.06, 0.09, 0.16))
    
    y -= 22
    # Authors Banner Box
    doc.set_fill_color(0.97, 0.98, 0.99)
    doc.set_stroke_color(0.85, 0.88, 0.92)
    doc.set_line_width(0.8)
    doc.rect(45, y - 68, doc.w - 90, 78, fill=True, stroke=True)
    
    # 4 Members Grid
    members = [
        ("Member A (Acquisition)", "Optical PPG (60 FPS) & IMU SCG (100 Hz)", 52),
        ("Member B (YOU - Core DSP)", "Conditioning, Systolic Peaks & PRV FFT", 180),
        ("Member C (ML Classifier)", "Dual-Target: Stress & Arrhythmia Risk", 310),
        ("Member D (Dynamics)", "Coupled Van der Pol Stress Twin Sim", 440)
    ]
    for title, desc, xpos in members:
        is_b = "Member B" in title
        col = (0.92, 0.35, 0.1) if is_b else (0.1, 0.15, 0.25)
        doc.draw_text(title, xpos, y - 8, 8, "Helvetica-Bold", col)
        doc.draw_text(desc[:26], xpos, y - 22, 7, "Helvetica", (0.35, 0.4, 0.45))
        if len(desc) > 26:
            doc.draw_text(desc[26:], xpos, y - 32, 7, "Helvetica", (0.35, 0.4, 0.45))
            
    y -= 95
    
    # Section 1: Executive Summary
    doc.draw_text("1. EXECUTIVE SUMMARY & RESEARCH OBJECTIVES", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1.5)
    doc.line(45, y - 4, 180, y - 4)
    y -= 18
    
    p1 = [
        "Cardiovascular autonomic dysfunction and cardiac arrhythmias represent leading global healthcare burdens.",
        "While traditional 12-lead Electrocardiography (ECG) remains the clinical gold standard, its reliance on adhesive electrodes",
        "and specialized analog instrumentation prevents continuous, frictionless everyday monitoring. This capstone engineering",
        "thesis proposes a complete Cyber-Physical Digital Twin software system implemented on standard consumer smartphones."
    ]
    for line in p1:
        doc.draw_text(line, 45, y, 8.5, "Helvetica", (0.2, 0.25, 0.3))
        y -= 12
        
    y -= 6
    p2 = [
        "The project introduces a dual-modal mobile acquisition framework eliminating all external hardware requirements:",
        "  • Primary Optical Contact PPG: 60 FPS camera video tracking capillary blood volume pulse illuminated by the LED flash.",
        "  • Secondary Mechanical Chest SCG: 100 Hz 3-axis MEMS accelerometer recording sternal cardiac recoil micro-vibrations.",
        "The extracted signals are conditioned using digital Butterworth filtering, processed via adaptive systolic peak detection,",
        "and converted to Pulse Rate Variability (PRV) biomarkers. A trained machine learning classifier predicts sympathovagal",
        "stress and arrhythmia risk, which parameterizes a coupled nonlinear oscillator simulating rhythm breakdown under stress."
    ]
    for line in p2:
        doc.draw_text(line, 45, y, 8.5, "Helvetica", (0.2, 0.25, 0.3))
        y -= 12
        
    y -= 12
    # Section 2: Architecture Table
    doc.draw_text("2. SYSTEM ARCHITECTURE & 4-STAGE PIPELINE CONTRACTS", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1.5)
    doc.line(45, y - 4, 180, y - 4)
    y -= 18
    
    # Table Header
    doc.set_fill_color(0.1, 0.15, 0.25)
    doc.rect(45, y - 14, doc.w - 90, 16, fill=True, stroke=False)
    headers = [("STAGE", 52), ("OWNER", 110), ("INPUT DATA", 175), ("CORE TECHNOLOGY", 295), ("OUTPUT CONTRACT", 445)]
    for htitle, hx in headers:
        doc.draw_text(htitle, hx, y - 10, 7.5, "Courier-Bold", (1, 1, 1))
        
    y -= 16
    rows = [
        ("Stage 1", "Member A", "Camera/IMU Sensor", "Flashlock, ROI Avg, IMU Polling", "RawSignal[t] @ 60/100Hz", False),
        ("Stage 2", "Member B (YOU)", "Raw Optical Series", "Butterworth, Peak Det, PRV FFT", "PRV Vector {SDNN, LF/HF}", True),
        ("Stage 3", "Member C", "PRV Feature Table", "XGBoost / Random Forest (WESAD)", "Risk Profile {Stress, AFib}", False),
        ("Stage 4", "Member D", "Risk Probabilities", "Coupled Van der Pol ODE Runge-Kutta", "Dynamic Phase Trajectory", False)
    ]
    for sname, owner, inp, tech, outp, focal in rows:
        fill_col = (1.0, 0.96, 0.93) if focal else (0.98, 0.98, 0.99) if "Stage 1" in sname or "Stage 3" in sname else (1, 1, 1)
        doc.set_fill_color(*fill_col)
        doc.rect(45, y - 14, doc.w - 90, 16, fill=True, stroke=False)
        doc.set_stroke_color(0.85, 0.88, 0.92)
        doc.set_line_width(0.5)
        doc.line(45, y - 14, doc.w - 90, y - 14)
        
        tcol = (0.92, 0.35, 0.1) if focal else (0.1, 0.15, 0.25)
        doc.draw_text(sname, 52, y - 10, 8, "Helvetica-Bold", tcol)
        doc.draw_text(owner, 110, y - 10, 7.5, "Helvetica", (0.2, 0.25, 0.3))
        doc.draw_text(inp, 175, y - 10, 7.5, "Helvetica", (0.2, 0.25, 0.3))
        doc.draw_text(tech, 295, y - 10, 7.5, "Helvetica", (0.2, 0.25, 0.3))
        doc.draw_text(outp, 445, y - 10, 7.5, "Courier-Bold" if focal else "Courier", tcol)
        y -= 16

    # ==========================================
    # PAGE 2: MEMBER A & MEMBER B (DSP DETAILS)
    # ==========================================
    doc.new_page()
    y = doc.h - 70
    
    doc.draw_text("3. METHODOLOGY & SUBSYSTEM SPECIFICATIONS", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1.5)
    doc.line(45, y - 4, 180, y - 4)
    y -= 18
    
    # Member A Section
    doc.draw_text("STAGE 1 (MEMBER A): MOBILE SENSOR INGESTION & QUALITY CONTROL", 45, y, 9.5, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 14
    m_a_desc = [
        "1. Optical Camera Pipeline: Flutter/React Native native bridge capturing uncompressed video frames at 60 FPS.",
        "   Locking exposure duration and ISO sensitivity while activating the rear torch LED to ensure constant skin illumination.",
        "2. Spatial Region of Interest (ROI): Extracting a centered 120x120 pixel bounding box from each frame and calculating the",
        "   mean green channel intensity:  I_G(t) = (1 / |ROI|) * sum(Pixel_G(x, y, t)).",
        "3. Contact Verification Heuristic: Analyzing the Red/Green channel ratio. Human fingertip transillumination exhibits",
        "   I_R / I_G >= 1.8. If the ratio drops, the system triggers a visual prompt notifying the user of poor optical contact.",
        "4. Sternal SCG Acceleration: Concurrently polling the device 3-axis accelerometer at 100 Hz via SENSOR_DELAY_FASTEST.",
        "   Extracting the dorso-ventral axis (a_z) representing perpendicular sternal recoil micro-displacements."
    ]
    for line in m_a_desc:
        doc.draw_text(line, 45, y, 8, "Helvetica", (0.2, 0.25, 0.3))
        y -= 11

    y -= 10
    # Member B Section (YOUR WORK - HIGHLIGHTED)
    doc.set_fill_color(1.0, 0.96, 0.93)
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1)
    doc.rect(45, y - 220, doc.w - 90, 225, fill=True, stroke=True)
    
    doc.draw_text("STAGE 2 (MEMBER B — YOUR ROLE): DIGITAL SIGNAL PROCESSING & PRV ENGINE", 52, y - 10, 9.5, "Helvetica-Bold", (0.92, 0.35, 0.1))
    y -= 24
    
    m_b_desc = [
        "As Member B, your responsibilities form the mathematical and signal processing foundation of the capstone thesis:",
        "",
        "A. Zero-Phase Digital Bandpass Filtering:",
        "   • PPG Channel: 4th-order zero-phase Butterworth bandpass filter (0.5 Hz - 5.0 Hz, corresponding to 30 - 300 bpm).",
        "     Removes low-frequency baseline drift (respiration, finger pressure) and high-frequency CMOS sensor noise.",
        "   • SCG Channel: 4th-order Butterworth bandpass filter (1.0 Hz - 20.0 Hz). Eliminates large thoracic respiratory",
        "     excursions (0.1 - 0.4 Hz) to isolate aortic valve opening (AO) and rapid ventricular ejection (RE) micro-vibrations.",
        "",
        "B. Adaptive Systolic Peak Detection & Outlier Rejection:",
        "   • Computes numerical gradient / derivative to amplify the rapid systolic upslope.",
        "   • Implements adaptive dual-thresholding with a 250 ms physiological blanking refractory period to prevent false",
        "     detection of dicrotic notches or reflected peripheral waves.",
        "   • Produces the discrete Peak-to-Peak Interval series: PPI[i] = t_peak[i] - t_peak[i-1] in milliseconds.",
        "   • Applies ectopic beat filtering: rejecting non-physiological intervals outside the 300 ms - 1800 ms window.",
        "",
        "C. Clinical Pulse Rate Variability (PRV) Feature Extraction:",
        "   • Time-Domain Metrics: Mean Heart Rate (bpm), SDNN (overall autonomic variability), RMSSD (parasympathetic",
        "     vagal tone: sqrt(mean(diff(PPI)^2))), and pNN50 (% of successive intervals differing by > 50 ms).",
        "   • Frequency-Domain Spectral Analysis (FFT): Uneven PPI series are resampled to 4.0 Hz via cubic spline interpolation.",
        "     Power Spectral Density (PSD) is estimated via Welch's periodogram, integrating specific clinical bands:",
        "       - Low-Frequency (LF: 0.04 - 0.15 Hz): Sympathetic & baroreflex modulation.",
        "       - High-Frequency (HF: 0.15 - 0.40 Hz): Parasympathetic vagal tone (respiratory sinus arrhythmia).",
        "       - LF/HF Ratio: The gold-standard clinical index of sympathovagal autonomic balance and acute stress."
    ]
    for line in m_b_desc:
        doc.draw_text(line, 52, y, 7.8, "Helvetica", (0.15, 0.2, 0.25))
        y -= 10.5
        
    y -= 225
    
    # Member C Section
    doc.draw_text("STAGE 3 (MEMBER C): DUAL-TARGET MACHINE LEARNING RISK PROFILING", 45, y, 9.5, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 14
    m_c_desc = [
        "1. Training Benchmarks: Utilizing WESAD (Wearable Stress and Affect Detection) and CapnoBase/MIMIC-III PPG benchmarks.",
        "2. Dual-Target Formulation:",
        "   • Target 1 (Autonomic Stress): Binary/3-class classifier (Baseline Rest vs. Cognitive/Physical Stress).",
        "   • Target 2 (Arrhythmia Screening): Regular Sinus Rhythm vs. Pulse Irregularity / Atrial Fibrillation.",
        "3. Algorithms & Rigor: Random Forest, Gradient Boosted Trees (XGBoost), and Logistic Regression evaluated via",
        "   Leave-One-Subject-Out (LOSO) cross-validation to strictly prevent patient data leakage. SHAP values explain features."
    ]
    for line in m_c_desc:
        doc.draw_text(line, 45, y, 8, "Helvetica", (0.2, 0.25, 0.3))
        y -= 11

    # ==========================================
    # PAGE 3: MEMBER D, DEFENSE STRATEGY & TIMELINE
    # ==========================================
    doc.new_page()
    y = doc.h - 70
    
    # Member D Section
    doc.draw_text("STAGE 4 (MEMBER D): COUPLED NONLINEAR OSCILLATOR HEART DIGITAL TWIN", 45, y, 9.5, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 14
    m_d_desc = [
        "1. Biophysical Foundation: Implements a coupled Van der Pol / FitzHugh-Nagumo dynamical pacemaker system representing",
        "   the Sinoatrial (SA) node pacemaker coupled to ventricular myocardium:",
        "       x''(t) - mu * (1 - x(t)^2) * x'(t) + omega^2 * x(t) = c12 * (y(t) - x(t)) + F_stress(t)",
        "2. Parameter Personalization: Member B's measured heart rate sets baseline frequency omega, while Member C's risk score",
        "   modulates the nonlinear damping factor mu and coupling coefficient c12.",
        "3. Stress Challenge Simulation: Under healthy profiles, the phase-space trajectory (x' vs x) forms a stable, periodic limit",
        "   cycle. Under high-risk profiles subjected to synthetic sympathetic drive F_stress(t), the model exhibits period-doubling",
        "   bifurcations, simulating paroxysmal tachycardia and chaotic arrhythmia breakdown in real-time visuals."
    ]
    for line in m_d_desc:
        doc.draw_text(line, 45, y, 8, "Helvetica", (0.2, 0.25, 0.3))
        y -= 11
        
    y -= 14
    
    # Section 4: Defense Strategy
    doc.draw_text("4. VIVA DEFENSE STRATEGY (ANTICIPATING COMMITTEE INQUIRIES)", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1.5)
    doc.line(45, y - 4, 180, y - 4)
    y -= 18
    
    qa_list = [
        ("Q1: 'Commercial smartphone apps already measure heart rate. What is your novel engineering contribution?'",
         ["A1: Commercial mobile apps are closed-source commercial utilities that output a single coarse number (heart rate bpm)",
          "using black-box filters. Our thesis develops a fully transparent, white-box pipeline extracting clinical Pulse Rate",
          "Variability (PRV) spectral densities, pairs it with an explainable ML autonomic risk classifier, and drives a real-time",
          "nonlinear cardiac dynamical digital twin to simulate physiological stress collapse. No commercial app does this."]),
        ("Q2: 'Is optical Pulse Rate Variability (PRV) clinically accepted as an equivalent to ECG HRV?'",
         ["A2: Extensive clinical literature (e.g. Lu et al., Schäfer et al.) proves that under quasi-stationary conditions, optical",
          "pulse-to-pulse intervals correlate with ECG RR intervals with R^2 >= 0.95. Time-domain RMSSD and spectral HF power",
          "exhibit near-perfect clinical equivalence. We explicitly model and account for Pulse Transit Time (PTT) variance."]),
        ("Q3: 'Why did the team include the Seismocardiography (SCG) phone-on-chest mode?'",
         ["A3: SCG captures cardiac mechanical recoil rather than peripheral blood volume. Supporting both modalities demonstrates",
          "sensor fusion, handling high-rate IMU interrupts, multi-axis orientation decoupling, and motion artifact filtering."])
    ]
    for q, ans_lines in qa_list:
        doc.draw_text(q, 45, y, 8, "Helvetica-Bold", (0.06, 0.09, 0.16))
        y -= 10
        for aline in ans_lines:
            doc.draw_text(aline, 55, y, 7.8, "Helvetica", (0.3, 0.35, 0.4))
            y -= 9.5
        y -= 4.5

    y -= 4
    
    # Section 5: Timeline & Milestones
    doc.draw_text("5. PROJECT TIMELINE & 16-WEEK CAPSTONE SCHEDULE", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    doc.set_stroke_color(0.92, 0.35, 0.1)
    doc.set_line_width(1.5)
    doc.line(45, y - 4, 180, y - 4)
    y -= 18
    
    # Milestone Table
    doc.set_fill_color(0.1, 0.15, 0.25)
    doc.rect(45, y - 14, doc.w - 90, 16, fill=True, stroke=False)
    doc.draw_text("PHASE & WEEKS", 52, y - 10, 7.5, "Courier-Bold", (1, 1, 1))
    doc.draw_text("KEY MILESTONE DELIVERABLES", 160, y - 10, 7.5, "Courier-Bold", (1, 1, 1))
    doc.draw_text("PRIMARY LEAD", 450, y - 10, 7.5, "Courier-Bold", (1, 1, 1))
    
    y -= 16
    schedule = [
        ("Phase 1 (W1-W4)", "Camera 60 FPS ROI extraction, Flashlock & 100 Hz IMU client", "Member A"),
        ("Phase 2 (W5-W8)", "Butterworth DSP, adaptive peak detector & PRV feature table", "Member B (YOU)"),
        ("Phase 3 (W9-W12)", "WESAD benchmark ML model & Van der Pol ODE twin simulation", "Members C & D"),
        ("Phase 4 (W13-W16)", "End-to-end integration, live unified UI, thesis report & defense", "All Members")
    ]
    for ph, deliv, lead in schedule:
        doc.set_fill_color(0.98, 0.98, 0.99)
        doc.rect(45, y - 14, doc.w - 90, 16, fill=True, stroke=False)
        doc.set_stroke_color(0.85, 0.88, 0.92)
        doc.set_line_width(0.5)
        doc.line(45, y - 14, doc.w - 90, y - 14)
        
        doc.draw_text(ph, 52, y - 10, 8, "Helvetica-Bold", (0.1, 0.15, 0.25))
        doc.draw_text(deliv, 160, y - 10, 7.5, "Helvetica", (0.2, 0.25, 0.3))
        doc.draw_text(lead, 450, y - 10, 7.5, "Courier-Bold" if "YOU" in lead else "Courier", (0.92, 0.35, 0.1) if "YOU" in lead else (0.2, 0.25, 0.3))
        y -= 16

    doc.build_pdf("thesis_proposal_report.pdf")

if __name__ == "__main__":
    generate_thesis_proposal()
