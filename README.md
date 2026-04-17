# 🍊 Mango Intelligent Receipt Agent (Typhoon Vision Edition)

โปรเจกต์แอปพลิเคชันเว็บ (Proof of Concept) สำหรับสกัดข้อมูลจากใบเสร็จรับเงินและใบกำกับภาษีอัตโนมัติ โดยใช้เทคโนโลยี **Vision AI** และ **Large Language Model (LLM)** ที่ออกแบบมาเพื่อทำงานร่วมกับระบบ ERP (Enterprise Resource Planning)

แอปพลิเคชันนี้ถูกพัฒนาด้วย **Streamlit** และใช้ขุมพลังจาก **OpenTyphoon API** (โมเดลภาษาและ Vision สัญชาติไทย) เพื่อแก้ปัญหาการกรอกข้อมูลบัญชีด้วยมือ (Manual Data Entry)

---

## ✨ Features (จุดเด่นของระบบ)

1. **Two-Step AI Pipeline (สถาปัตยกรรม 2 ขั้นตอน)**
   - **Step 1 (Vision):** ใช้ `typhoon-ocr` สกัดข้อความและตารางจากรูปภาพให้ออกมาเป็น Markdown ที่สะอาดและแม่นยำ
   - **Step 2 (Reasoning):** ใช้ `typhoon-v2.5-30b-a3b-instruct` ทำหน้าที่เป็น "สมองนักบัญชี" จัดโครงสร้างข้อความ (Unstructured Data) ให้อยู่ในรูปแบบ JSON (Structured Data)
2. **Image Quality Gate (ด่านตรวจคุณภาพภาพ)**
   - ผสานการทำงานกับ `OpenCV` เพื่อคำนวณค่าความคมชัด (Laplacian Variance) ดักจับภาพเบลอก่อนส่งเข้า AI เพื่อป้องกันปัญหา *Garbage In, Garbage Out* และประหยัดโควตา API
3. **Automated ERP Auditing (ระบบตรวจสอบบัญชีอัตโนมัติ)**
   - มีระบบ Data Cleansing ล้างอักขระขยะออกจากเลขผู้เสียภาษี
   - Cross-check ข้อมูลกับ Mock ERP Database ทันที เพื่อตรวจสอบว่า:
     - บิลนี้ออกในชื่อบริษัทเราถูกต้องหรือไม่?
     - Vendor (ผู้ขาย) มีตัวตนในระบบหรือไม่?
     - ยอดเงิน (Grand Total) ตรงกับใบสั่งซื้อ (PO) หรือไม่?
4. **Human-in-the-Loop (HITL) & Export**
   - พนักงานบัญชีสามารถตรวจสอบและแก้ไขข้อมูล (Editable Data Grid) ก่อนบันทึกจริง
   - บันทึกข้อมูลสะสมลงใน Session State และรองรับการ Export เป็นไฟล์ `.csv` (รองรับภาษาไทย `utf-8-sig`) สำหรับนำไปใช้งานต่อใน Excel

---

## 🛠️ Tech Stack

- **Frontend / UI:** [Streamlit](https://streamlit.io/)
- **Computer Vision:** OpenCV (`cv2`), Numpy
- **Data Manipulation:** Pandas
- **AI / LLM:** OpenTyphoon API (`typhoon-ocr`, `typhoon-v2.5-30b-a3b-instruct`)
- **Environment Management:** `python-dotenv`

---

## 🚀 วิธีการติดตั้งและใช้งาน (Installation & Setup)

**1. Clone โปรเจกต์**
```bash
git clone [https://github.com/your-username/mango-ai-agent-ocr.git](https://github.com/your-username/mango-ai-agent-ocr.git)
cd mango-ai-agent-ocr
```

**2. ติดตั้ง Library ที่จำเป็น**
แนะนำให้สร้าง Virtual Environment ก่อน แล้วรันคำสั่ง:
```bash
pip install streamlit opencv-python numpy pandas requests python-dotenv
```

**3. ตั้งค่า Environment Variables**
สร้างไฟล์ `.env` ไว้ในโฟลเดอร์เดียวกับ `app.py` และใส่ API Key ของ OpenTyphoon:
```env
GENAI_TYPHON_API_KEY=your_typhoon_api_key_here
```

**4. รันแอปพลิเคชัน**
```bash
streamlit run app.py
```

---

## 📂 โครงสร้างโปรเจกต์ (Project Structure)
```text
mango-ai-agent-ocr/
│
├── app.py               # โค้ดหลักของแอปพลิเคชัน (Streamlit UI + Logic)
├── .env                 # ไฟล์เก็บ API Key (ไม่ต้องอัปโหลดขึ้น GitHub)
├── .gitignore           # ซ่อนไฟล์ที่ไม่ต้องการอัปโหลด
└── README.md            # เอกสารอธิบายโปรเจกต์
```

---

## 💡 System Workflow (ขั้นตอนการทำงานหลังบ้าน)

1. **Upload:** ผู้ใช้อัปโหลดรูปภาพใบเสร็จ (`.jpg`, `.png`)
2. **Pre-processing:** `OpenCV` แปลงภาพเป็น Grayscale และคำนวณค่า Focus Measure ถ้าน้อยกว่า 50 จะตีกลับให้ถ่ายใหม่
3. **OCR Extraction:** ส่งภาพไปที่ `https://api.opentyphoon.ai/v1/ocr` คืนค่ากลับมาเป็น Markdown Table
4. **LLM Formatting:** ส่งข้อความ Markdown เข้าสู่ `typhoon-v2.5-30b` พร้อม System Prompt บังคับ Output เป็น JSON Schema
5. **Validation:** นำ JSON มาลบช่องว่าง (Data Cleansing) และจับคู่กับตัวแปร `MOCK_ERP`
6. **Approval & Export:** แสดงผลให้ผู้ใช้กดยืนยัน และรวบรวมเป็น DataFrame เพื่อดาวน์โหลดเป็น CSV

---
*Developed for Educational & Proof of Concept Purposes.*
```

---
