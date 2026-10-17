#!/usr/bin/env python3
"""
compile_approval_report_pdf.py
Pure-Python standalone PDF generator that compiles advisor_approval_report.tex
into a publication-quality multi-page PDF: advisor_approval_report.pdf
"""

import sys
import os

class PDFReportGenerator:
    def __init__(self, page_width=595.28, page_height=841.89): # A4
        self.w = page_width
        self.h = page_height
        self.pages = []
        self.current_page_ops = []
        self.page_number = 0
        
    def new_page(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []
        self.page_number += 1
        self._add_decorations()

    def _add_decorations(self):
        # Header hairline
        self.set_stroke_color(0.8, 0.85, 0.9)
        self.set_line_width(0.7)
        self.line(45, self.h - 42, self.w - 45, self.h - 42)
        
        # Header text
        self.draw_text("DEPARTMENT OF COMPUTER ENGINEERING • CAPSTONE THESIS PROPOSAL", 45, self.h - 35, 7.5, "Helvetica-Bold", (0.35, 0.4, 0.45))
        self.draw_text("ACADEMIC YEAR 2026-2027", self.w - 150, self.h - 35, 7.5, "Helvetica", (0.35, 0.4, 0.45))
        
        # Footer hairline
        self.line(45, 42, self.w - 45, 42)
        self.draw_text("CONFIDENTIAL • FACULTY ADVISOR APPROVAL DOSSIER", 45, 32, 7.5, "Helvetica", (0.45, 0.5, 0.55))
        self.draw_text(f"Page {self.page_number}", self.w - 75, 32, 7.5, "Helvetica-Bold", (0.2, 0.25, 0.3))

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
            '•': '-', '–': '-', '—': '--', '’': "'", '‘': "'",
            '“': '"', '”': '"', '…': '...', '°': ' deg', 'µ': 'u',
            'λ': 'lambda', '≤': '<=', '≥': '>=', '≈': '~', '→': '->',
            '✓': '[x]', '×': 'x'
        }
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = text.encode('ascii', errors='replace').decode('ascii')
        return text.replace('\\', '\\\\').replace('(', '\\(').replace(')', '\\)')

    def draw_text(self, text, x, y, size=9.5, font="Helvetica", color=(0.1, 0.15, 0.25)):
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

    def finalize(self):
        if self.current_page_ops:
            self.pages.append(self.current_page_ops)
            self.current_page_ops = []

    def save_pdf(self, filename):
        self.finalize()
        objects = []
        def add_obj(data):
            objects.append(data)
            return len(objects)

        cat_id = add_obj("<< /Type /Catalog /Pages 3 0 R >>")
        outlines_id = add_obj("<< /Type /Outlines /Count 0 >>")
        pages_id = add_obj("") # placeholder

        f_helv = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
        f_helv_b = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>")
        f_helv_o = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Oblique >>")
        f_cour = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier >>")
        f_cour_b = add_obj("<< /Type /Font /Subtype /Type1 /BaseFont /Courier-Bold >>")

        page_refs = []
        for p_ops in self.pages:
            c_stream = "\n".join(p_ops)
            c_bytes = c_stream.encode('latin1')
            c_id = add_obj(f"<< /Length {len(c_bytes)} >>\nstream\n{c_stream}\nendstream")
            
            p_obj = (
                f"<< /Type /Page\n"
                f"/Parent {pages_id} 0 R\n"
                f"/MediaBox [0 0 {self.w:.2f} {self.h:.2f}]\n"
                f"/Contents {c_id} 0 R\n"
                f"/Resources << /Font <<\n"
                f"  /Helvetica {f_helv} 0 R\n"
                f"  /Helvetica-Bold {f_helv_b} 0 R\n"
                f"  /Helvetica-Oblique {f_helv_o} 0 R\n"
                f"  /Courier {f_cour} 0 R\n"
                f"  /Courier-Bold {f_cour_b} 0 R\n"
                f">> >> >>"
            )
            p_id = add_obj(p_obj)
            page_refs.append(f"{p_id} 0 R")

        kids = " ".join(page_refs)
        objects[pages_id - 1] = f"<< /Type /Pages /Kids [ {kids} ] /Count {len(page_refs)} >>"

        with open(filename, "wb") as f:
            f.write(b"%PDF-1.4\n")
            offsets = []
            for i, obj in enumerate(objects):
                offsets.append(f.tell())
                f.write(f"{i+1} 0 obj\n{obj}\nendobj\n".encode('latin1'))
            xref_pos = f.tell()
            f.write(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode('latin1'))
            for off in offsets:
                f.write(f"{off:010d} 00000 n \n".encode('latin1'))
            f.write(f"trailer\n<< /Size {len(objects) + 1}\n/Root {cat_id} 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode('latin1'))
        print(f"Successfully generated {filename} ({len(self.pages)} pages)")


def build_report():
    pdf = PDFReportGenerator()
    
    # ==============================================================
    # PAGE 1: TITLE, OBJECTIVE & SCOPE PART 1
    # ==============================================================
    pdf.new_page()
    y = pdf.h - 68
    
    # Title badge
    pdf.set_fill_color(0.92, 0.35, 0.1) # Accent orange
    pdf.rect(45, y, 105, 13, fill=True, stroke=False)
    pdf.draw_text("ADVISOR APPROVAL DOSSIER", 49, y + 3, 6.8, "Courier-Bold", (1, 1, 1))
    
    y -= 22
    pdf.draw_text("A Multi-Modal Mobile Sensing System for Real-Time Cardiac", 45, y, 14.5, "Helvetica-Bold", (0.06, 0.09, 0.16))
    y -= 17
    pdf.draw_text("Autonomic Risk Profiling and In-Silico Dynamical Rhythm Stress Simulation", 45, y, 14.5, "Helvetica-Bold", (0.06, 0.09, 0.16))
    
    y -= 15
    pdf.draw_text("Department of Computer Engineering - Senior Capstone Design Project (4 Members)", 45, y, 8.5, "Helvetica-Oblique", (0.35, 0.4, 0.45))
    
    # Top metadata line
    pdf.set_stroke_color(0.85, 0.88, 0.92)
    pdf.set_line_width(0.8)
    pdf.line(45, y - 8, pdf.w - 45, y - 8)
    y -= 24
    
    # SECTION 1: OBJECTIVE
    pdf.draw_text("SECTION 1: OBJECTIVES", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    pdf.draw_text("(Formulated strictly with measurable action verbs)", 195, y + 1, 8, "Helvetica-Oblique", (0.4, 0.45, 0.5))
    pdf.set_stroke_color(0.92, 0.35, 0.1)
    pdf.set_line_width(1.2)
    pdf.line(45, y - 3, 175, y - 3)
    y -= 16
    
    objectives = [
        ("1. To develop", "an end-to-end, multi-stage mobile health and cyber-physical digital-twin pipeline that captures non-invasive physiological cardiac signals directly via standard smartphone sensors, performs digital filtering, and extracts clinically validated Pulse Rate Variability (PRV) metrics in real time."),
        ("2. To identify", "and validate mechanical cardiac micro-vibrations along the sternal dorso-ventral axis using smartphone 3-axis MEMS accelerometers (Seismocardiography - SCG), while establishing an automated contingency fallback to optical camera-based photoplethysmography (Contact PPG) and gold-standard clinical databases to guarantee system robustness."),
        ("3. To improve", "the interpretability and utility of consumer-grade mobile cardiovascular metrics by training a dual-target supervised machine learning model (evaluating acute autonomic stress and arrhythmia screening) that outputs an explainable risk vector rather than a closed black-box pulse count."),
        ("4. To mitigate", "the gap between static diagnostic health metrics and predictive physiological outcomes by implementing a parameterized, coupled nonlinear cardiac oscillator (Van der Pol / FitzHugh-Nagumo model) that dynamically simulates cardiac rhythm stability, period-doubling bifurcations, and arrhythmic breakdown under simulated autonomic stress challenges.")
    ]
    for tag, desc in objectives:
        pdf.draw_text(tag, 45, y, 8.5, "Helvetica-Bold", (0.92, 0.35, 0.1))
        words = desc.split()
        line = ""
        first = True
        for w in words:
            if len(line + " " + w) > 92:
                pdf.draw_text(line, 115 if first else 115, y, 8, "Helvetica", (0.2, 0.25, 0.3))
                y -= 10
                line = w
                first = False
            else:
                line = (line + " " + w).strip()
        if line:
            pdf.draw_text(line, 115 if first else 115, y, 8, "Helvetica", (0.2, 0.25, 0.3))
            y -= 10
        y -= 4

    y -= 10
    # SECTION 2: SCOPE
    pdf.draw_text("SECTION 2: PROJECT SCOPE & BOUNDARY SPECIFICATIONS", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    pdf.set_stroke_color(0.92, 0.35, 0.1)
    pdf.set_line_width(1.2)
    pdf.line(45, y - 3, 260, y - 3)
    y -= 16
    
    scope_points = [
        ("1. What you do or do not consider in this project:", [
            "• Inclusions: Acquisition of cardiac signals using built-in smartphone sensors: primary evaluation of Seismocardiography (SCG) via 3-axis MEMS accelerometer placed flat on the sternum during resting postures; secondary fallback to optical contact PPG via the rear camera and LED flashlight; tertiary fallback to public clinical benchmark data (PhysioNet MIT-BIH, WESAD, CapnoBase). Zero-phase Butterworth filtering, systolic peak detection, time/frequency PRV feature extraction, dual-target ML classification, and coupled Van der Pol ODE rhythm simulation.",
            "• Exclusions: Invasive cardiovascular monitoring (arterial lines), high-motion ambulatory exercise testing (running motion artifacts overwhelm unanchored smartphone sensors), and multi-lead diagnostic ECG morphology analysis (e.g. ST-segment elevation for acute infarction)."
        ]),
        ("2. What you cannot do or exclude from this project:", [
            "• Non-diagnostic Disclaimer: The software cannot provide medical diagnoses, clinical prescriptions, or formal therapeutic interventions. It is strictly an engineering research and decision-support prototype.",
            "• Regulatory Certification: The project cannot achieve FDA/CE medical device hardware certification within the academic timeframe.",
            "• Demographic Boundaries: Excludes pediatric and neonatal populations; validation is restricted to adult physiological ranges."
        ])
    ]
    for stitle, slines in scope_points:
        pdf.draw_text(stitle, 45, y, 8.5, "Helvetica-Bold", (0.1, 0.15, 0.25))
        y -= 11
        for sline in slines:
            words = sline.split()
            cur = ""
            for w in words:
                if len(cur + " " + w) > 105:
                    pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
                    y -= 9.5
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
                y -= 9.5
            y -= 2
        y -= 4

    # ==============================================================
    # PAGE 2: SCOPE PART 2 & BACKGROUND PART 1
    # ==============================================================
    pdf.new_page()
    y = pdf.h - 65
    
    scope_part2 = [
        ("3. Limitations of the Approach:", [
            "• Sensor Noise Floors: Consumer-grade smartphone MEMS accelerometers exhibit intrinsic thermal noise and variable quantization drift compared to clinical-grade seismometers.",
            "• Posture Dependence: Sternal SCG acquisition strictly requires supine (lying flat) or semi-recumbent (45 deg) resting postures to isolate cardiac micro-g accelerations from gravitational reorientation.",
            "• Optical Contact Sensitivity: The camera PPG fallback is sensitive to excessive fingertip pressure (which blanches capillary beds) and skin tone absorption variations.",
            "• Mobile OS Timestamp Jitter: Mobile operating systems (Android/iOS) introduce non-deterministic thread scheduling and variable inter-frame timestamp jitter that must be compensated for via cubic-spline interpolation."
        ]),
        ("4. Ranges of Data, Date, and Time Parameters:", [
            "• Heart Rate Operational Range: 40 bpm to 180 bpm (fundamental cardiac frequency: 0.67 Hz to 3.0 Hz).",
            "• Peak-to-Peak Interval (PPI) Window: 333 ms to 1500 ms (physiological bounds for outlier rejection).",
            "• Sensor Sampling Rates: 100 Hz to 200 Hz for IMU Accelerometer; 30 FPS or 60 FPS for camera PPG; resampled to 4.0 Hz uniform grid for spectral PRV analysis.",
            "• Measurement Durations: 30 seconds for instantaneous pulse rate; 60 to 120 seconds for short-term statistical and spectral PRV (SDNN, RMSSD, LF/HF).",
            "• Academic Schedule: 16 weeks across two semesters (Fall 2026 - Spring 2027)."
        ]),
        ("5. Platform(s) and Device(s) Used Specifically:", [
            "• Mobile Sensing Hardware: Commercially available iOS (iPhone with CoreMotion/AVFoundation) and Android (SensorManager/Camera2 API) devices containing 3-axis MEMS accelerometers and rear LED flash units.",
            "• Mobile Application Layer: Cross-platform Flutter / React Native client providing native camera/IMU hardware bindings and local WebSockets.",
            "• Computation Engine: Python 3.9+ backend (FastAPI, WebSockets, NumPy, SciPy, Scikit-Learn, XGBoost).",
            "• Simulation Environment: Python Runge-Kutta ODE numerical solvers (scipy.integrate.solve_ivp RK45) with validation against MATLAB/Simulink biophysical models."
        ])
    ]
    for stitle, slines in scope_part2:
        pdf.draw_text(stitle, 45, y, 8.5, "Helvetica-Bold", (0.1, 0.15, 0.25))
        y -= 11
        for sline in slines:
            words = sline.split()
            cur = ""
            for w in words:
                if len(cur + " " + w) > 105:
                    pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
                    y -= 9.5
                    cur = w
                else:
                    cur = (cur + " " + w).strip()
            if cur:
                pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
                y -= 9.5
            y -= 2
        y -= 4

    y -= 10
    # SECTION 3: BACKGROUND
    pdf.draw_text("SECTION 3: BACKGROUND & LITERATURE REVIEW", 45, y, 11, "Helvetica-Bold", (0.06, 0.09, 0.16))
    pdf.set_stroke_color(0.92, 0.35, 0.1)
    pdf.set_line_width(1.2)
    pdf.line(45, y - 3, 240, y - 3)
    y -= 16
    
    pdf.draw_text("3.1 Problem Statement & Healthcare Importance", 45, y, 9, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 11
    bg_p1 = [
        "Cardiovascular diseases (CVDs) remain the leading contributor to global mortality, causing an estimated 17.9 million deaths",
        "annually. Concurrently, chronic psychophysiological stress and autonomic nervous system (ANS) dysregulation--characterized",
        "by prolonged sympathetic hyperactivity and suppressed parasympathetic vagal tone--is a recognized precursor to hypertension,",
        "fatal arrhythmias, and ischemic heart failure. While conventional 12-lead Electrocardiography (ECG) and continuous Holter",
        "monitors represent the clinical gold standard, their reliance on specialized analog differential amplifiers, high-impedance",
        "instrumentation circuitry, skin-abrasive conductive gel, and adhesive electrodes creates severe logistical friction.",
        "Consequently, cardiac risk monitoring remains episodic and hospital-bound. There is an urgent engineering imperative for",
        "frictionless, zero-external-hardware cardiovascular monitoring systems that leverage ubiquitous mobile consumer devices."
    ]
    for line in bg_p1:
        pdf.draw_text(line, 45, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
        y -= 9.8

    # ==============================================================
    # PAGE 3: LITERATURE REVIEW & GAPS
    # ==============================================================
    pdf.new_page()
    y = pdf.h - 65
    
    pdf.draw_text("3.2 Completed Outline of Literature Review", 45, y, 9, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 12
    
    lit_review = [
        ("1. Optical Contact Photoplethysmography (PPG):",
         "Measures localized blood volume pulse oscillations through transillumination of peripheral capillary beds, governed by the Beer-Lambert law of optical absorption. Hertzman (1938) initially demonstrated optical pulse detection, which was formalized for consumer digital CMOS camera sensors by Verkruysse et al. (2008) and Jonathan & Leahy (2010). Modern studies confirm that green light (520-540 nm) provides the highest signal-to-noise ratio (SNR) due to the distinct absorption peak of oxyhemoglobin (HbO2)."),
        ("2. Mechanical Sternal Seismocardiography (SCG):",
         "Records mechanical micro-vibrations of the chest wall induced by left ventricular myocardial contraction, isovolumetric momentum transfer, and rapid blood ejection into the ascending aorta. First investigated by Baevskii et al. (1964), modern MEMS sensors have revitalized SCG. Inan et al. (2015) and Tadi et al. (2017) established that standard smartphone MEMS accelerometers resting flat on the sternum capture aortic valve opening (AO) and rapid ejection (RE) micro-vibrations within 1.0-20.0 Hz, providing an acoustic-mechanical sensing pathway without optical hardware."),
        ("3. Pulse Rate Variability (PRV) as an Autonomic Surrogate:",
         "Clinical standards for Heart Rate Variability (HRV) defined by the Task Force of the European Society of Cardiology and NASPE (1996) identify SDNN, RMSSD, and spectral LF/HF ratios as primary indicators of sympathovagal balance. Extensive clinical validation by Lu et al. (2009) and Schafer & Vagedes (2013) demonstrated that in stationary resting conditions, peak-to-peak intervals (PPI) derived from optical PPG and mechanical SCG correlate with ECG-derived RR intervals with high statistical fidelity (R^2 >= 0.95)."),
        ("4. Nonlinear Cardiac Dynamical Modeling:",
         "The human cardiac conduction system operates as an assembly of nonlinear, self-oscillating pacemakers. Van der Pol & Van der Mark (1928) originally formulated coupled relaxation oscillators to model Sinoatrial (SA) and Atrioventricular (AV) nodal dynamics. FitzHugh (1961) and Glass (2001) demonstrated that cardiac conduction anomalies, ventricular flutter, and chaotic fibrillation are nonlinear dynamical state bifurcations triggered when intrinsic oscillator damping and coupling strengths are perturbed by external autonomic/sympathetic stress drive.")
    ]
    for ltitle, ldesc in lit_review:
        pdf.draw_text(ltitle, 45, y, 8.2, "Helvetica-Bold", (0.1, 0.15, 0.25))
        y -= 10
        words = ldesc.split()
        cur = ""
        for w in words:
            if len(cur + " " + w) > 105:
                pdf.draw_text(cur, 55, y, 7.6, "Helvetica", (0.25, 0.3, 0.35))
                y -= 9.2
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            pdf.draw_text(cur, 55, y, 7.6, "Helvetica", (0.25, 0.3, 0.35))
            y -= 9.2
        y -= 3

    y -= 8
    pdf.draw_text("3.3 Identified Research Gaps", 45, y, 9, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 12
    
    gaps = [
        ("• Gap 1: The Disconnect Between Static Measurement and Predictive Dynamic Simulation.",
         "Commercial mobile health apps function purely as passive, closed-loop meters: they output a single static integer (e.g. '72 bpm') or a proprietary wellness score. None of these platforms bridge measured physiological parameters into an in-silico dynamical twin capable of simulating how that specific individual's cardiac rhythm stability will evolve or break down under simulated autonomic stress challenges."),
        ("• Gap 2: Lack of Transparent, Open-Source Multi-Modal White-Box Architectures.",
         "Existing mobile heart rate applications rely on proprietary, closed-source black-box heuristics that cannot be peer-reviewed, audited, or academically validated. Furthermore, they rely strictly on a single sensor modality (camera only), rendering them brittle under hardware constraints and vulnerable to optical artifacts."),
        ("• Gap 3: Sensor Underutilization and High Hardware Barriers in Capstone Projects.",
         "Previous student engineering capstones frequently rely on standalone external biomedical circuit breakout boards (e.g., AD8232 ECG boards), introducing physical connection failures and cumbersome hardware maintenance while leaving the advanced, high-precision MEMS inertial sensors built directly into modern smartphones completely underutilized.")
    ]
    for gtitle, gdesc in gaps:
        pdf.draw_text(gtitle, 45, y, 8.2, "Helvetica-Bold", (0.92, 0.35, 0.1))
        y -= 10
        words = gdesc.split()
        cur = ""
        for w in words:
            if len(cur + " " + w) > 105:
                pdf.draw_text(cur, 55, y, 7.6, "Helvetica", (0.25, 0.3, 0.35))
                y -= 9.2
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            pdf.draw_text(cur, 55, y, 7.6, "Helvetica", (0.25, 0.3, 0.35))
            y -= 9.2
        y -= 3

    # ==============================================================
    # PAGE 4: RESEARCH QUESTIONS, CONTRIBUTIONS & SIGN-OFF
    # ==============================================================
    pdf.new_page()
    y = pdf.h - 65
    
    pdf.draw_text("3.4 Research Questions", 45, y, 9, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 12
    
    rqs = [
        ("Research Question 1 (Sensing):", "Can a standard smartphone's embedded 3-axis MEMS accelerometer reliably capture sternal mechanical recoil micro-vibrations (SCG) with sufficient signal-to-noise ratio to yield mathematically valid beat-to-beat intervals under resting conditions?"),
        ("Research Question 2 (Robustness):", "How can an automated multi-tier sensing fallback architecture (Sternal SCG -> Camera PPG -> Benchmarked Clinical Datasets) be structured to ensure zero failure states during real-time deployment?"),
        ("Research Question 3 (Simulation):", "How can empirical Pulse Rate Variability features extracted from mobile sensors be mapped mathematically into the damping and coupling parameters of a coupled nonlinear cardiac oscillator to simulate visual rhythm destabilization and chaotic bifurcations under autonomic stress?")
    ]
    for rtag, rdesc in rqs:
        pdf.draw_text(rtag, 45, y, 8.2, "Helvetica-Bold", (0.1, 0.15, 0.25))
        y -= 10
        words = rdesc.split()
        cur = ""
        for w in words:
            if len(cur + " " + w) > 105:
                pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
                y -= 9.5
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.25, 0.3, 0.35))
            y -= 9.5
        y -= 4

    y -= 10
    pdf.draw_text("3.5 Academic & Engineering Contributions", 45, y, 9, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 12
    
    contribs = [
        "1. A fully unified, transparent Cyber-Physical Digital Twin software pipeline combining mobile signal acquisition, zero-phase DSP, machine learning risk classification, and nonlinear dynamical ODE simulation.",
        "2. A novel comparative evaluation between mechanical sternal SCG and optical fingertip PPG on consumer smartphone hardware.",
        "3. An open, reproducible benchmark validating mobile Pulse Rate Variability (PRV) against established clinical datasets (WESAD and CapnoBase/MIMIC-III)."
    ]
    for c in contribs:
        words = c.split()
        cur = ""
        for w in words:
            if len(cur + " " + w) > 105:
                pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.2, 0.25, 0.3))
                y -= 9.5
                cur = w
            else:
                cur = (cur + " " + w).strip()
        if cur:
            pdf.draw_text(cur, 55, y, 7.8, "Helvetica", (0.2, 0.25, 0.3))
            y -= 9.5
        y -= 4

    y -= 25
    # Team & Faculty Sign-off Box
    pdf.set_fill_color(0.97, 0.98, 0.99)
    pdf.set_stroke_color(0.85, 0.88, 0.92)
    pdf.set_line_width(0.8)
    pdf.rect(45, y - 110, pdf.w - 90, 115, fill=True, stroke=True)
    
    pdf.draw_text("CAPSTONE PROJECT APPROVAL & ENDORSEMENT", 55, y - 12, 8.5, "Helvetica-Bold", (0.06, 0.09, 0.16))
    y -= 26
    
    # 4 Members
    pdf.draw_text("Student Members (Computer Engineering):", 55, y, 8, "Helvetica-Bold", (0.1, 0.15, 0.25))
    pdf.draw_text("Faculty Advisor Review:", 320, y, 8, "Helvetica-Bold", (0.1, 0.15, 0.25))
    y -= 14
    
    pdf.draw_text("Member A: _____________________________ [Acquisition]", 55, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    pdf.draw_text("Advisor Signature: _______________________", 320, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    y -= 14
    pdf.draw_text("Member B: _____________________________ [Core DSP]", 55, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    pdf.draw_text("Date: ______________  Status: [ ] APPROVED", 320, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    y -= 14
    pdf.draw_text("Member C: _____________________________ [ML Model]", 55, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    pdf.draw_text("Comments / Revisions Requested:", 320, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    y -= 14
    pdf.draw_text("Member D: _____________________________ [Dynamics]", 55, y, 7.5, "Courier", (0.35, 0.4, 0.45))
    pdf.draw_text("[  ] Conditional Approval   [  ] Rejected", 320, y, 7.5, "Courier", (0.35, 0.4, 0.45))

    pdf.save_pdf("advisor_approval_report.pdf")

if __name__ == "__main__":
    build_report()
